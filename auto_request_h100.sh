#!/bin/bash

REGION="eu-west-2"
INSTANCE_TYPE="p5.4xlarge"
CHECK_INTERVAL=300
LOG_FILE="h100_request_log.txt"

echo "=== Auto H100 Instance Requester ===" | tee -a "$LOG_FILE"
echo "Region: $REGION" | tee -a "$LOG_FILE"
echo "Instance Type: $INSTANCE_TYPE (H100)" | tee -a "$LOG_FILE"
echo "Check Interval: $CHECK_INTERVAL seconds (15 minutes)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

EXISTING_KEY=$(ls -t gpu-instance-key-*.pem 2>/dev/null | head -n 1)

if [ -n "$EXISTING_KEY" ]; then
    KEY_NAME="${EXISTING_KEY%.pem}"
    KEY_FILE="$EXISTING_KEY"
    echo "Using existing key: $KEY_FILE" | tee -a "$LOG_FILE"
    
    aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" &>/dev/null
    
    if [ $? -ne 0 ]; then
        echo "Key not found in AWS, creating new key..." | tee -a "$LOG_FILE"
        TIMESTAMP=$(date +%Y%m%d-%H%M%S)
        KEY_NAME="gpu-instance-key-${TIMESTAMP}"
        KEY_FILE="${KEY_NAME}.pem"
        
        aws ec2 create-key-pair \
            --region "$REGION" \
            --key-name "$KEY_NAME" \
            --query 'KeyMaterial' \
            --output text > "$KEY_FILE"
        
        chmod 400 "$KEY_FILE"
        echo "✓ Key pair created: $KEY_FILE" | tee -a "$LOG_FILE"
    fi
else
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    KEY_NAME="gpu-instance-key-${TIMESTAMP}"
    KEY_FILE="${KEY_NAME}.pem"
    
    echo "Creating new key pair: $KEY_NAME" | tee -a "$LOG_FILE"
    aws ec2 create-key-pair \
        --region "$REGION" \
        --key-name "$KEY_NAME" \
        --query 'KeyMaterial' \
        --output text > "$KEY_FILE"
    
    chmod 400 "$KEY_FILE"
    echo "✓ Key pair created: $KEY_FILE" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"

EXISTING_SG=$(aws ec2 describe-security-groups \
    --region "$REGION" \
    --filters "Name=group-name,Values=gpu-instance-sg-*" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null || echo "")

if [ -n "$EXISTING_SG" ] && [ "$EXISTING_SG" != "None" ]; then
    SECURITY_GROUP="$EXISTING_SG"
    echo "Using existing security group: $SECURITY_GROUP" | tee -a "$LOG_FILE"
else
    echo "Creating security group..." | tee -a "$LOG_FILE"
    SG_TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    SG_NAME="gpu-instance-sg-${SG_TIMESTAMP}"
    SECURITY_GROUP=$(aws ec2 create-security-group \
        --region "$REGION" \
        --group-name "$SG_NAME" \
        --description "Security group for GPU instance ${SG_TIMESTAMP}" \
        --query 'GroupId' \
        --output text)
    
    echo "✓ Security group created: $SECURITY_GROUP" | tee -a "$LOG_FILE"
    
    MY_IP=$(curl -s https://checkip.amazonaws.com)
    aws ec2 authorize-security-group-ingress \
        --region "$REGION" \
        --group-id "$SECURITY_GROUP" \
        --protocol tcp \
        --port 22 \
        --cidr "${MY_IP}/32" > /dev/null
    
    echo "✓ SSH access allowed from: $MY_IP" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Getting AMI..." | tee -a "$LOG_FILE"

AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8* (Ubuntu 24.04)*" "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].[ImageId,Name]' \
    --output text)

if [ -z "$AMI_ID" ] || [ "$AMI_ID" == "None" ]; then
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 24.04)*" "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].[ImageId,Name]' \
        --output text)
fi

if [ -z "$AMI_ID" ] || [ "$AMI_ID" == "None" ]; then
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04)*" "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].[ImageId,Name]' \
        --output text)
fi

AMI_ID_ONLY=$(echo "$AMI_ID" | awk '{print $1}')
AMI_NAME=$(echo "$AMI_ID" | cut -f2-)

echo "Using AMI: $AMI_ID_ONLY" | tee -a "$LOG_FILE"
if [ -n "$AMI_NAME" ]; then
    echo "AMI Name: $AMI_NAME" | tee -a "$LOG_FILE"
fi

AMI_ID="$AMI_ID_ONLY"
DISK_SIZE=500

echo "" | tee -a "$LOG_FILE"
echo "Starting automatic instance request loop..." | tee -a "$LOG_FILE"
echo "Press Ctrl+C to stop" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

attempt=0

while true; do
    attempt=$((attempt + 1))
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] Attempt #$attempt: Requesting $INSTANCE_TYPE..." | tee -a "$LOG_FILE"
    
    RESULT=$(aws ec2 run-instances \
        --region "$REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SECURITY_GROUP" \
        --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${DISK_SIZE},VolumeType=gp3}" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=Auto-H100-${timestamp}}]" \
        --output json 2>&1)
    
    if echo "$RESULT" | grep -q "InsufficientInstanceCapacity"; then
        echo "[$timestamp] ✗ Insufficient capacity - will retry in 15 minutes" | tee -a "$LOG_FILE"
    elif echo "$RESULT" | grep -q "InstanceLimitExceeded"; then
        echo "[$timestamp] ✗ Instance limit exceeded - check your quota" | tee -a "$LOG_FILE"
    elif echo "$RESULT" | grep -q "Unsupported"; then
        echo "[$timestamp] ✗ Instance type not supported in this region/AZ" | tee -a "$LOG_FILE"
    elif echo "$RESULT" | grep -q "error\|Error"; then
        echo "[$timestamp] ✗ Error occurred:" | tee -a "$LOG_FILE"
        echo "$RESULT" | tee -a "$LOG_FILE"
    else
        INSTANCE_ID=$(echo "$RESULT" | jq -r '.Instances[0].InstanceId' 2>/dev/null)
        
        if [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "null" ]; then
            echo "[$timestamp] ✓ SUCCESS! Instance launched: $INSTANCE_ID" | tee -a "$LOG_FILE"
            echo "" | tee -a "$LOG_FILE"
            
            echo "Waiting for public IP..." | tee -a "$LOG_FILE"
            sleep 5
            
            PUBLIC_IP=""
            for i in {1..30}; do
                PUBLIC_IP=$(aws ec2 describe-instances \
                    --region "$REGION" \
                    --instance-ids "$INSTANCE_ID" \
                    --query 'Reservations[0].Instances[0].PublicIpAddress' \
                    --output text)
                
                if [ -n "$PUBLIC_IP" ] && [ "$PUBLIC_IP" != "None" ]; then
                    break
                fi
                sleep 2
            done
            
            echo "" | tee -a "$LOG_FILE"
            echo "==========================================" | tee -a "$LOG_FILE"
            echo "H100 INSTANCE SUCCESSFULLY DEPLOYED!" | tee -a "$LOG_FILE"
            echo "==========================================" | tee -a "$LOG_FILE"
            echo "" | tee -a "$LOG_FILE"
            echo "Instance ID:    $INSTANCE_ID" | tee -a "$LOG_FILE"
            echo "Instance Type:  $INSTANCE_TYPE" | tee -a "$LOG_FILE"
            echo "GPU:            NVIDIA H100" | tee -a "$LOG_FILE"
            echo "Public IP:      $PUBLIC_IP" | tee -a "$LOG_FILE"
            echo "Key File:       $KEY_FILE" | tee -a "$LOG_FILE"
            echo "Security Group: $SECURITY_GROUP" | tee -a "$LOG_FILE"
            echo "" | tee -a "$LOG_FILE"
            echo "==========================================" | tee -a "$LOG_FILE"
            echo "CONNECTION INSTRUCTIONS" | tee -a "$LOG_FILE"
            echo "==========================================" | tee -a "$LOG_FILE"
            echo "" | tee -a "$LOG_FILE"
            
            if [ -n "$PUBLIC_IP" ] && [ "$PUBLIC_IP" != "None" ]; then
                echo "Wait ~2 minutes for instance to boot, then connect:" | tee -a "$LOG_FILE"
                echo "" | tee -a "$LOG_FILE"
                echo "  ssh -i $KEY_FILE ubuntu@$PUBLIC_IP" | tee -a "$LOG_FILE"
            fi
            
            echo "" | tee -a "$LOG_FILE"
            echo "==========================================" | tee -a "$LOG_FILE"
            echo "INSTANCE MANAGEMENT" | tee -a "$LOG_FILE"
            echo "==========================================" | tee -a "$LOG_FILE"
            echo "" | tee -a "$LOG_FILE"
            echo "Terminate instance:" | tee -a "$LOG_FILE"
            echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION" | tee -a "$LOG_FILE"
            echo "" | tee -a "$LOG_FILE"
            echo "==========================================" | tee -a "$LOG_FILE"
            
            exit 0
        else
            echo "[$timestamp] ✗ Unknown error - could not parse instance ID" | tee -a "$LOG_FILE"
        fi
    fi
    
    echo "[$timestamp] Sleeping for 15 minutes..." | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    sleep "$CHECK_INTERVAL"
done

