import * as cdk from "aws-cdk-lib/core";
import { Construct } from "constructs/lib/construct";
import * as ecr from "aws-cdk-lib/aws-ecr";
import { BaseStackProps } from "../types";

export class EcrStack extends cdk.Stack {
  readonly repository: ecr.Repository;
  readonly repositoryUri: string;

  constructor(scope: Construct, id: string, props: BaseStackProps) {
    super(scope, id, props);

    this.repository = new ecr.Repository(this, `${props.appName}-EcrRepo`, {
      repositoryName: `${props.appName.toLowerCase()}-agent`,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      emptyOnDelete: false,
      lifecycleRules: [
        {
          description: "Keep last 10 images",
          maxImageCount: 10,
          rulePriority: 1,
          tagStatus: ecr.TagStatus.ANY,
        },
      ],
      imageScanOnPush: true,
    });

    this.repositoryUri = this.repository.repositoryUri;

    new cdk.CfnOutput(this, "EcrRepositoryUri", {
      value: this.repositoryUri,
      description: "ECR Repository URI",
      exportName: `${props.appName}-EcrRepositoryUri`,
    });

    new cdk.CfnOutput(this, "EcrRepositoryName", {
      value: this.repository.repositoryName,
      description: "ECR Repository Name",
      exportName: `${props.appName}-EcrRepositoryName`,
    });
  }
}
