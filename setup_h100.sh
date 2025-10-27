#!/bin/bash

set -e

echo "=== H100 Instance Setup Script ==="
echo ""

echo "Step 1: Creating SSH key pair..."
aws ec2 create-key-pair --key-name h100-key --query 'KeyMaterial' --output text > h100-key.pem
chmod 400 h100-key.pem
echo "✓ SSH key created: h100-key.pem"
echo ""

echo "Step 2: Configuring security group for SSH access..."
aws ec2 authorize-security-group-ingress \
  --group-id sg-0c4c987ed10a3747c \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0
echo "✓ Security group configured"
echo ""

echo "Step 3: Finding GPU-optimized AMI..."
AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*" "Name=state,Values=available" \
  --query "Images[0].ImageId" \
  --output text)
echo "✓ Found AMI: $AMI_ID"
echo ""

echo "Step 4: Launching H100 instance..."
INSTANCE_OUTPUT=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type p5.4xlarge \
  --key-name h100-key \
  --security-group-ids sg-0c4c987ed10a3747c \
  --subnet-id subnet-04a694f82c0e2f9b3 \
  --associate-public-ip-address \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=H100-GPU-Instance}]')

INSTANCE_ID=$(echo "$INSTANCE_OUTPUT" | jq -r '.Instances[0].InstanceId')
echo "✓ Instance launched: $INSTANCE_ID"
echo ""

echo "Step 5: Waiting for instance to be running (this may take 1-2 minutes)..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
echo "✓ Instance is running"
echo ""

echo "Step 6: Retrieving instance details..."
INSTANCE_INFO=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query "Reservations[0].Instances[0].{PublicIpAddress:PublicIpAddress,PrivateIpAddress:PrivateIpAddress,InstanceType:InstanceType}")
echo "$INSTANCE_INFO"
echo ""

PUBLIC_IP=$(echo "$INSTANCE_INFO" | jq -r '.PublicIpAddress')
echo "=== Setup Complete ==="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "Private Key: h100-key.pem"
echo ""
echo "To connect:"
echo "  ssh -i h100-key.pem ubuntu@$PUBLIC_IP"
echo ""
echo "To verify GPU:"
echo "  nvidia-smi"
echo "  python3 -c \"import torch; print(torch.cuda.is_available())\""
echo ""
echo "Cost: ~\$3.06/hour"
echo "To stop: aws ec2 stop-instances --instance-ids $INSTANCE_ID"
echo "To terminate: aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
