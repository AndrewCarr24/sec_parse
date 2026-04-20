#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { BaseStackProps } from "../lib/types";
import { EcrStack, AgentCoreStack } from "../lib/stacks";

const app = new cdk.App();

const APP_NAME = "ragAgent";

const existingImageUri = app.node.tryGetContext("imageUri") as
  | string
  | undefined;

const deploymentProps: BaseStackProps = {
  appName: APP_NAME,
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || "us-east-1",
  },
};

const ecrStack = new EcrStack(app, `${APP_NAME}-EcrStack`, deploymentProps);

const imageUri = existingImageUri || `${ecrStack.repositoryUri}:latest`;

if (existingImageUri) {
  console.log(`Using provided image URI: ${existingImageUri}`);
} else {
  console.log(`Using default ECR image URI: ${imageUri}`);
}

const agentCoreStack = new AgentCoreStack(app, `${APP_NAME}-AgentCoreStack`, {
  ...deploymentProps,
  imageUri: imageUri,
});

agentCoreStack.addDependency(ecrStack);
