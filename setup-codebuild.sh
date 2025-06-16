#!/bin/bash

# Setup AWS CodeBuild project for building container images
# Usage: ./setup-codebuild.sh [PROJECT_NAME] [GITHUB_REPO_URL]

set -e

PROJECT_NAME=${1:-"mistral-7b-container-build"}
GITHUB_REPO=${2:-"https://github.com/jicowan/mistral-7b-kubernetes.git"}
AWS_REGION=${AWS_REGION:-"us-west-2"}

echo "🏗️  Setting up AWS CodeBuild project"
echo "===================================="
echo "Project Name: $PROJECT_NAME"
echo "GitHub Repo: $GITHUB_REPO"
echo "AWS Region: $AWS_REGION"

# Create CodeBuild service role
echo ""
echo "1. Creating CodeBuild service role..."

ROLE_NAME="${PROJECT_NAME}-codebuild-role"
POLICY_NAME="${PROJECT_NAME}-codebuild-policy"

# Create trust policy
cat > codebuild-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "codebuild.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create the role
aws iam create-role \
    --role-name $ROLE_NAME \
    --assume-role-policy-document file://codebuild-trust-policy.json \
    --region $AWS_REGION 2>/dev/null || echo "Role already exists"

# Create policy for CodeBuild
cat > codebuild-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:GetAuthorizationToken",
        "ecr:CreateRepository",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "sts:GetCallerIdentity"
      ],
      "Resource": "*"
    }
  ]
}
EOF

# Attach policy to role
aws iam put-role-policy \
    --role-name $ROLE_NAME \
    --policy-name $POLICY_NAME \
    --policy-document file://codebuild-policy.json

# Get role ARN
ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text)
echo "   ✅ Role created: $ROLE_ARN"

# Create CodeBuild project
echo ""
echo "2. Creating CodeBuild project..."

cat > codebuild-project.json << EOF
{
  "name": "$PROJECT_NAME",
  "description": "Build container images for Mistral 7B inference server",
  "source": {
    "type": "GITHUB",
    "location": "$GITHUB_REPO",
    "buildspec": "buildspec.yml"
  },
  "artifacts": {
    "type": "NO_ARTIFACTS"
  },
  "environment": {
    "type": "LINUX_CONTAINER",
    "image": "aws/codebuild/standard:7.0",
    "computeType": "BUILD_GENERAL1_LARGE",
    "privilegedMode": true
  },
  "serviceRole": "$ROLE_ARN",
  "timeoutInMinutes": 120
}
EOF

aws codebuild create-project --cli-input-json file://codebuild-project.json --region $AWS_REGION

echo "   ✅ CodeBuild project created: $PROJECT_NAME"

# Clean up temporary files
rm -f codebuild-trust-policy.json codebuild-policy.json codebuild-project.json

echo ""
echo "3. Setup complete!"
echo "=================="
echo "CodeBuild Project: $PROJECT_NAME"
echo "Service Role: $ROLE_ARN"
echo ""
echo "To start a build:"
echo "  aws codebuild start-build --project-name $PROJECT_NAME --region $AWS_REGION"
echo ""
echo "To monitor the build:"
echo "  1. Go to AWS Console > CodeBuild > Build projects > $PROJECT_NAME"
echo "  2. Or use: aws codebuild batch-get-builds --ids <build-id>"
echo ""
echo "Expected build time: 30-60 minutes"
echo "Expected cost: \$3-8 per build (BUILD_GENERAL1_LARGE)"
