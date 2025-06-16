#!/bin/bash

# Launch EC2 instance for building container images
# Usage: ./setup-ec2-builder.sh

set -e

INSTANCE_TYPE="c6i.4xlarge"  # 16 vCPUs, 32GB RAM
AMI_ID="ami-0c02fb55956c7d316"  # Amazon Linux 2023 (update for your region)
KEY_NAME="your-key-pair"  # Replace with your key pair name
SECURITY_GROUP="sg-xxxxxxxxx"  # Replace with your security group
SUBNET_ID="subnet-xxxxxxxxx"  # Replace with your subnet ID

echo "🚀 Launching EC2 instance for container builds"
echo "=============================================="
echo "Instance Type: $INSTANCE_TYPE"
echo "AMI ID: $AMI_ID"

# User data script to setup Docker and AWS CLI
cat > user-data.sh << 'EOF'
#!/bin/bash
yum update -y
yum install -y docker git

# Start Docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

echo "Setup complete! Ready for container builds."
EOF

# Launch instance
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --count 1 \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SECURITY_GROUP \
    --subnet-id $SUBNET_ID \
    --user-data file://user-data.sh \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=mistral-container-builder}]' \
    --iam-instance-profile Name=EC2-ECR-Role \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "✅ Instance launched: $INSTANCE_ID"
echo ""
echo "Wait for instance to be ready, then:"
echo "1. ssh -i ~/.ssh/$KEY_NAME.pem ec2-user@<instance-ip>"
echo "2. git clone https://github.com/jicowan/mistral-7b-kubernetes.git"
echo "3. cd mistral-7b-kubernetes"
echo "4. ./build-all-images.sh"

# Clean up
rm user-data.sh
EOF
