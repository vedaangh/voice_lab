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
echo "Finding latest NVIDIA Deep Learning AMIs..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

CONDA_AMI=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning Proprietary Nvidia Driver AMI GPU PyTorch*Ubuntu*" "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].[ImageId,Name]' \
    --output text)

OSS_AMI=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 24.04)*" "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].[ImageId,Name]' \
    --output text)

declare -a AVAILABLE_AMIS
declare -A AMI_NAMES

if [ -n "$CONDA_AMI" ] && [ "$CONDA_AMI" != "None" ]; then
    CONDA_AMI_ID=$(echo "$CONDA_AMI" | awk '{print $1}')
    CONDA_AMI_NAME=$(echo "$CONDA_AMI" | cut -f2-)
    AVAILABLE_AMIS+=("$CONDA_AMI_ID")
    AMI_NAMES["$CONDA_AMI_ID"]="$CONDA_AMI_NAME"
fi

if [ -n "$OSS_AMI" ] && [ "$OSS_AMI" != "None" ]; then
    OSS_AMI_ID=$(echo "$OSS_AMI" | awk '{print $1}')
    OSS_AMI_NAME=$(echo "$OSS_AMI" | cut -f2-)
    AVAILABLE_AMIS+=("$OSS_AMI_ID")
    AMI_NAMES["$OSS_AMI_ID"]="$OSS_AMI_NAME"
fi

if [ ${#AVAILABLE_AMIS[@]} -eq 0 ]; then
    echo "Error: No suitable AMIs found in region $REGION" | tee -a "$LOG_FILE"
    exit 1
fi

echo "=== Available AMIs ===" | tee -a "$LOG_FILE"
for i in "${!AVAILABLE_AMIS[@]}"; do
    ami_id="${AVAILABLE_AMIS[$i]}"
    ami_name="${AMI_NAMES[$ami_id]}"
    echo "$((i+1)). $ami_id" | tee -a "$LOG_FILE"
    echo "   $ami_name" | tee -a "$LOG_FILE"
    
    if [[ "$ami_name" == *"Proprietary"* ]]; then
        echo "   Type: Conda (source activate pytorch)" | tee -a "$LOG_FILE"
        echo "   ✓ Full conda environments with 'source activate pytorch'" | tee -a "$LOG_FILE"
    else
        echo "   Type: OSS (system python - may need setup)" | tee -a "$LOG_FILE"
    fi
    
    if [[ "$ami_name" == *"PyTorch 2.8"* ]] || [[ "$ami_name" == *"PyTorch 2.9"* ]] || [[ "$ami_name" == *"PyTorch 3."* ]]; then
        echo "   ✓ PyTorch 2.8+ - torchcodec AudioDecoder supported" | tee -a "$LOG_FILE"
    fi
    echo "" | tee -a "$LOG_FILE"
done

read -p "Select AMI number (or press Enter for #1): " ami_selection

if [ -z "$ami_selection" ]; then
    ami_selection=1
fi

if ! [[ "$ami_selection" =~ ^[0-9]+$ ]] || [ "$ami_selection" -lt 1 ] || [ "$ami_selection" -gt ${#AVAILABLE_AMIS[@]} ]; then
    echo "Error: Invalid AMI selection" | tee -a "$LOG_FILE"
    exit 1
fi

AMI_ID="${AVAILABLE_AMIS[$((ami_selection-1))]}"
AMI_NAME="${AMI_NAMES[$AMI_ID]}"

echo "" | tee -a "$LOG_FILE"
echo "Selected AMI: $AMI_ID" | tee -a "$LOG_FILE"
echo "Name: $AMI_NAME" | tee -a "$LOG_FILE"
DISK_SIZE=500

echo "Disk Size: ${DISK_SIZE} GB" | tee -a "$LOG_FILE"
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

