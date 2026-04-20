import * as cdk from "aws-cdk-lib/core";
import { Construct } from "constructs/lib/construct";
import * as bedrockagentcore from "aws-cdk-lib/aws-bedrockagentcore";
import * as iam from "aws-cdk-lib/aws-iam";
import * as logs from "aws-cdk-lib/aws-logs";
import { BaseStackProps } from "../types";

export interface AgentCoreStackProps extends BaseStackProps {
  imageUri: string;
}

export class AgentCoreStack extends cdk.Stack {
  readonly agentCoreRuntime: bedrockagentcore.CfnRuntime;
  readonly agentCoreMemory: bedrockagentcore.CfnMemory;

  constructor(scope: Construct, id: string, props: AgentCoreStackProps) {
    super(scope, id, props);

    const region = cdk.Stack.of(this).region;
    const accountId = cdk.Stack.of(this).account;

    /*****************************
     * AgentCore Memory
     ******************************/

    this.agentCoreMemory = new bedrockagentcore.CfnMemory(
      this,
      `${props.appName}-AgentCoreMemory`,
      {
        name: `${props.appName}_Memory`,
        eventExpiryDuration: 30,
        description: `${props.appName} Memory resource with 30 days event expiry`,
        memoryStrategies: [
          {
            userPreferenceMemoryStrategy: {
              name: "user_preference_strategy",
              namespaces: ["/users/{actorId}/preferences"],
            },
          },
          {
            semanticMemoryStrategy: {
              name: "semantic_strategy",
              namespaces: ["/conversations/{actorId}/facts"],
            },
          },
          {
            summaryMemoryStrategy: {
              name: "summary_strategy",
              namespaces: ["/conversations/{sessionId}/summaries"],
            },
          },
        ],
      }
    );

    /*****************************
     * AgentCore Runtime
     ******************************/

    const runtimePolicy = new iam.PolicyDocument({
      statements: [
        new iam.PolicyStatement({
          sid: "ECRImageAccess",
          effect: iam.Effect.ALLOW,
          actions: ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
          resources: [`arn:aws:ecr:${region}:${accountId}:repository/*`],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ["logs:DescribeLogStreams", "logs:CreateLogGroup"],
          resources: [
            `arn:aws:logs:${region}:${accountId}:log-group:/aws/bedrock-agentcore/runtimes/*`,
          ],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ["logs:DescribeLogGroups"],
          resources: [`arn:aws:logs:${region}:${accountId}:log-group:*`],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ["logs:CreateLogStream", "logs:PutLogEvents"],
          resources: [
            `arn:aws:logs:${region}:${accountId}:log-group:/aws/bedrock-agentcore/runtimes/*:log-stream:*`,
          ],
        }),
        new iam.PolicyStatement({
          sid: "ECRTokenAccess",
          effect: iam.Effect.ALLOW,
          actions: ["ecr:GetAuthorizationToken"],
          resources: ["*"],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: [
            "xray:PutTraceSegments",
            "xray:PutTelemetryRecords",
            "xray:GetSamplingRules",
            "xray:GetSamplingTargets",
          ],
          resources: ["*"],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ["cloudwatch:PutMetricData"],
          resources: ["*"],
          conditions: {
            StringEquals: { "cloudwatch:namespace": "bedrock-agentcore" },
          },
        }),
        new iam.PolicyStatement({
          sid: "OTLPCloudWatchExport",
          effect: iam.Effect.ALLOW,
          actions: [
            "logs:PutLogEvents",
            "logs:CreateLogStream",
            "logs:CreateLogGroup",
            "logs:DescribeLogStreams",
          ],
          resources: [
            `arn:aws:logs:${region}:${accountId}:log-group:/aws/vendedlogs/bedrock-agentcore/*`,
            `arn:aws:logs:${region}:${accountId}:log-group:/aws/vendedlogs/bedrock-agentcore/*:log-stream:*`,
            `arn:aws:logs:${region}:${accountId}:log-group:aws/spans:*`,
          ],
        }),
        new iam.PolicyStatement({
          sid: "GetAgentAccessToken",
          effect: iam.Effect.ALLOW,
          actions: [
            "bedrock-agentcore:GetWorkloadAccessToken",
            "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
            "bedrock-agentcore:GetWorkloadAccessTokenForUserId",
          ],
          resources: [
            `arn:aws:bedrock-agentcore:${region}:${accountId}:workload-identity-directory/default`,
            `arn:aws:bedrock-agentcore:${region}:${accountId}:workload-identity-directory/default/workload-identity/agentName-*`,
          ],
        }),
        new iam.PolicyStatement({
          sid: "BedrockModelInvocation",
          effect: iam.Effect.ALLOW,
          actions: [
            "bedrock:InvokeModel",
            "bedrock:InvokeModelWithResponseStream",
          ],
          resources: [
            `arn:aws:bedrock:*::foundation-model/*`,
            `arn:aws:bedrock:${region}:${accountId}:*`,
          ],
        }),
        new iam.PolicyStatement({
          sid: "BedrockRuntimeOperations",
          effect: iam.Effect.ALLOW,
          actions: ["bedrock:Converse", "bedrock:ConverseStream"],
          resources: [`arn:aws:bedrock:*::foundation-model/*`],
        }),
        // AgentCore Memory operations
        new iam.PolicyStatement({
          sid: "BedrockAgentCoreMemory",
          effect: iam.Effect.ALLOW,
          actions: ["bedrock-agentcore:*"],
          resources: [
            `arn:aws:bedrock-agentcore:${region}:${accountId}:memory/*`,
          ],
        }),
        // S3 access for embedding model download
        new iam.PolicyStatement({
          sid: "S3ModelAccess",
          effect: iam.Effect.ALLOW,
          actions: [
            "s3:GetObject",
            "s3:ListBucket",
            "s3:HeadBucket",
          ],
          resources: [
            "arn:aws:s3:::us-east-1-sbxopsdatascience-sagemaker-data",
            "arn:aws:s3:::us-east-1-sbxopsdatascience-sagemaker-data/andrewcarr/hf_models/*",
          ],
        }),
        // CloudWatch Logs Delivery
        new iam.PolicyStatement({
          sid: "CloudWatchLogsDelivery",
          effect: iam.Effect.ALLOW,
          actions: [
            "logs:PutDeliverySource",
            "logs:PutDeliveryDestination",
            "logs:CreateDelivery",
            "logs:GetDeliverySource",
            "logs:GetDeliveryDestination",
            "logs:GetDelivery",
            "logs:DeleteDeliverySource",
            "logs:DeleteDeliveryDestination",
            "logs:DeleteDelivery",
          ],
          resources: ["*"],
        }),
      ],
    });

    const runtimeRole = new iam.Role(
      this,
      `${props.appName}-AgentCoreRuntimeRole`,
      {
        assumedBy: new iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
        description: "IAM role for RAG Agent AgentCore Runtime",
        inlinePolicies: {
          RuntimeAccessPolicy: runtimePolicy,
        },
      }
    );

    this.agentCoreRuntime = new bedrockagentcore.CfnRuntime(
      this,
      `${props.appName}-AgentCoreRuntime`,
      {
        agentRuntimeArtifact: {
          containerConfiguration: {
            containerUri: props.imageUri,
          },
        },
        agentRuntimeName: `${props.appName}_Agent`,
        protocolConfiguration: "HTTP",
        networkConfiguration: {
          networkMode: "PUBLIC",
        },
        roleArn: runtimeRole.roleArn,
        environmentVariables: {
          AWS_REGION: region,
          MEMORY_ID: this.agentCoreMemory.attrMemoryId,
          EMBEDDING_PROVIDER: "bedrock",

          // OpenTelemetry observability
          AGENT_OBSERVABILITY_ENABLED: "true",
          OTEL_SERVICE_NAME: `${props.appName}-agent`,
          OTEL_PYTHON_DISTRO: "aws_distro",
          OTEL_PYTHON_CONFIGURATOR: "aws_configurator",
          OTEL_EXPORTER_OTLP_PROTOCOL: "http/protobuf",
          OTEL_TRACES_EXPORTER: "otlp",
          OTEL_METRICS_EXPORTER: "otlp",
          OTEL_LOGS_EXPORTER: "otlp",
          OTEL_EXPORTER_OTLP_ENDPOINT: `https://xray.${region}.amazonaws.com`,
          OTEL_PROPAGATORS: "xray,tracecontext,baggage",
          OTEL_RESOURCE_ATTRIBUTES: `service.name=${props.appName}-agent,aws.log.group.names=/aws/bedrock-agentcore/runtimes/${props.appName}-agent,cloud.region=${region}`,
          OTEL_EXPORTER_OTLP_LOGS_HEADERS: `x-aws-log-group=/aws/bedrock-agentcore/runtimes/${props.appName}-agent,x-aws-log-stream=runtime-logs,x-aws-metric-namespace=bedrock-agentcore`,
        },
      }
    );

    // X-Ray resource policy for CloudWatch Logs
    new logs.CfnResourcePolicy(this, `${props.appName}-XRayResourcePolicy`, {
      policyName: `${props.appName}-XRayCloudWatchLogsAccess`,
      policyDocument: JSON.stringify({
        Version: "2012-10-17",
        Statement: [
          {
            Effect: "Allow",
            Principal: { Service: "xray.amazonaws.com" },
            Action: ["logs:PutLogEvents", "logs:CreateLogStream"],
            Resource: `arn:aws:logs:${region}:${accountId}:log-group:aws/spans:*`,
          },
        ],
      }),
    });

    /*****************************
     * Stack Outputs
     ******************************/

    new cdk.CfnOutput(this, "MemoryId", {
      value: this.agentCoreMemory.attrMemoryId,
      description: "AgentCore Memory ID",
      exportName: `${props.appName}-MemoryId`,
    });

    new cdk.CfnOutput(this, "RuntimeId", {
      value: this.agentCoreRuntime.attrAgentRuntimeId,
      description: "AgentCore Runtime ID",
      exportName: `${props.appName}-RuntimeId`,
    });

    new cdk.CfnOutput(this, "RuntimeArn", {
      value: this.agentCoreRuntime.attrAgentRuntimeArn,
      description: "AgentCore Runtime ARN",
      exportName: `${props.appName}-RuntimeArn`,
    });

    new cdk.CfnOutput(this, "RuntimeRoleArn", {
      value: runtimeRole.roleArn,
      description: "AgentCore Runtime Role ARN",
      exportName: `${props.appName}-RuntimeRoleArn`,
    });
  }
}
