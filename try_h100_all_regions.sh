#!/bin/bash

# AWS regions to try for H100 (p5 instances)
REGIONS=(
    "us-east-1"
    "us-east-2" 
    "us-west-2"
    "eu-west-2"
    "eu-west-1"
    "eu-central-1"
    "ap-northeast-1"
    "ap-southeast-1"
    "ap-south-1"
    "ca-central-1"
)

# H100 instance types to try
INSTANCE_TYPES=("p5.48xlarge" "p5.4xlarge")

KEY_NAME="gpu-instance-key-20251112-172151"
AMI_NAME="Deep Learning Proprietary Nvidia Driver AMI GPU PyTorch 2.3.1 (Ubuntu 20.04)*"

echo "=== Trying to launch H100 (p5) instance across AWS regions ==="
echo ""

for REGION in "${REGIONS[@]}"; do
    echo ">>> Trying region: $REGION"
    
    # Find AMI in this region
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters "Name=name,Values=$AMI_NAME" "Name=state,Values=available" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text 2>/dev/null)
    
    if [ -z "$AMI_ID" ] || [ "$AMI_ID" == "None" ]; then
        echo "  No suitable AMI found in $REGION, skipping..."
        continue
    fi
    echo "  Found AMI: $AMI_ID"
    
    # Check if key exists in this region
    KEY_EXISTS=$(aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" 2>/dev/null | grep -c "$KEY_NAME" || echo "0")
    if [ "$KEY_EXISTS" -eq 0 ]; then
        echo "  Key $KEY_NAME not found in $REGION, skipping..."
        continue
    fi
    
    # Get default VPC and subnet
    VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text 2>/dev/null)
    SUBNET_ID=$(aws ec2 describe-subnets --region "$REGION" --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[0].SubnetId' --output text 2>/dev/null)
    
    if [ -z "$SUBNET_ID" ] || [ "$SUBNET_ID" == "None" ]; then
        echo "  No default subnet in $REGION, skipping..."
        continue
    fi
    
    # Create or get security group
    SG_ID=$(aws ec2 describe-security-groups --region "$REGION" --filters "Name=group-name,Values=h100-ssh-access" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)
    if [ -z "$SG_ID" ] || [ "$SG_ID" == "None" ]; then
        echo "  Creating security group..."
        SG_ID=$(aws ec2 create-security-group --region "$REGION" --group-name "h100-ssh-access" --description "SSH access for H100" --vpc-id "$VPC_ID" --query 'GroupId' --output text 2>/dev/null)
        MY_IP=$(curl -s ifconfig.me)
        aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" --protocol tcp --port 22 --cidr "${MY_IP}/32" 2>/dev/null
    fi
    
    for INSTANCE_TYPE in "${INSTANCE_TYPES[@]}"; do
        echo "  Trying $INSTANCE_TYPE..."
        
        # Try spot first
        echo "    Trying spot instance..."
        SPOT_RESULT=$(aws ec2 run-instances \
            --region "$REGION" \
            --image-id "$AMI_ID" \
            --instance-type "$INSTANCE_TYPE" \
            --key-name "$KEY_NAME" \
            --security-group-ids "$SG_ID" \
            --subnet-id "$SUBNET_ID" \
            --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time}' \
            --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=h100-spot-$REGION}]" \
            --query 'Instances[0].InstanceId' \
            --output text 2>&1)
        
        if [[ "$SPOT_RESULT" != *"error"* ]] && [[ "$SPOT_RESULT" != *"Error"* ]] && [[ "$SPOT_RESULT" != *"capacity"* ]] && [ -n "$SPOT_RESULT" ] && [ "$SPOT_RESULT" != "None" ]; then
            echo ""
            echo "=== SUCCESS! Launched SPOT $INSTANCE_TYPE in $REGION ==="
            echo "Instance ID: $SPOT_RESULT"
            
            # Wait for public IP
            sleep 10
            PUBLIC_IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$SPOT_RESULT" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null)
            echo "Public IP: $PUBLIC_IP"
            echo ""
            echo "SSH command: ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
            exit 0
        fi
        echo "    Spot failed: $SPOT_RESULT"
        
        # Try on-demand
        echo "    Trying on-demand instance..."
        OD_RESULT=$(aws ec2 run-instances \
            --region "$REGION" \
            --image-id "$AMI_ID" \
            --instance-type "$INSTANCE_TYPE" \
            --key-name "$KEY_NAME" \
            --security-group-ids "$SG_ID" \
            --subnet-id "$SUBNET_ID" \
            --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=h100-ondemand-$REGION}]" \
            --query 'Instances[0].InstanceId' \
            --output text 2>&1)
        
        if [[ "$OD_RESULT" != *"error"* ]] && [[ "$OD_RESULT" != *"Error"* ]] && [[ "$OD_RESULT" != *"capacity"* ]] && [ -n "$OD_RESULT" ] && [ "$OD_RESULT" != "None" ]; then
            echo ""
            echo "=== SUCCESS! Launched ON-DEMAND $INSTANCE_TYPE in $REGION ==="
            echo "Instance ID: $OD_RESULT"
            
            # Wait for public IP
            sleep 10
            PUBLIC_IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$OD_RESULT" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null)
            echo "Public IP: $PUBLIC_IP"
            echo ""
            echo "SSH command: ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
            exit 0
        fi
        echo "    On-demand failed: $OD_RESULT"
    done
    echo ""
done

echo ""
echo "=== Failed to launch H100 in any region ==="
exit 1
