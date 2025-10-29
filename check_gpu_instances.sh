#!/bin/bash

set -e

echo "=== AWS GPU Instance Checker and Deployer ==="
echo ""

DEFAULT_REGION=$(aws configure get region)
if [ -z "$DEFAULT_REGION" ]; then
    DEFAULT_REGION="us-east-1"
fi

echo "Default region: $DEFAULT_REGION"
echo ""
echo "Common GPU regions:"
echo "  us-east-1      (N. Virginia)"
echo "  us-east-2      (Ohio)"
echo "  us-west-2      (Oregon)"
echo "  eu-west-1      (Ireland)"
echo "  eu-west-2      (London)"
echo "  eu-central-1   (Frankfurt)"
echo "  ap-southeast-1 (Singapore)"
echo "  ap-northeast-1 (Tokyo)"
echo ""
read -p "Enter region (or press Enter for $DEFAULT_REGION): " REGION

if [ -z "$REGION" ]; then
    REGION="$DEFAULT_REGION"
fi

echo ""
echo "Using region: $REGION"
echo ""

GPU_INSTANCE_FAMILIES=("g4dn" "g5" "g6" "p3" "p4d" "p4de" "p5")

declare -a AVAILABLE_INSTANCES
declare -A GPU_INFO
declare -A FAMILY_QUOTAS

get_quota_code() {
    local family=$1
    case $family in
        g4dn|g5|g6)
            echo "L-DB2E81BA"
            ;;
        p3|p4d|p4de|p5)
            echo "L-417A185B"
            ;;
        *)
            echo "L-DB2E81BA"
            ;;
    esac
}

echo "Checking GPU instances..."
echo ""

for family in "${GPU_INSTANCE_FAMILIES[@]}"; do
    echo "Checking $family family..."
    
    quota_code=$(get_quota_code "$family")
    
    if [ -z "${FAMILY_QUOTAS[$quota_code]}" ]; then
        quota=$(aws service-quotas get-service-quota \
            --region "$REGION" \
            --service-code ec2 \
            --quota-code "$quota_code" \
            --query 'Quota.Value' \
            --output text 2>/dev/null || echo "0")
        FAMILY_QUOTAS[$quota_code]=$quota
    else
        quota=${FAMILY_QUOTAS[$quota_code]}
    fi
    
    quota_int=$(echo "$quota" | awk '{print int($1)}')
    
    if [ "$quota_int" -eq 0 ]; then
        echo "  No quota for $family family (quota: $quota vCPUs)"
        continue
    fi
    
    echo "  Quota for $family family: $quota_int vCPUs"
    
    instance_types=$(aws ec2 describe-instance-types \
        --region "$REGION" \
        --filters "Name=instance-type,Values=${family}.*" \
        --query 'InstanceTypes[*].[InstanceType,VCpuInfo.DefaultVCpus]' \
        --output text 2>/dev/null || echo "")
    
    if [ -z "$instance_types" ]; then
        echo "  No $family instances available in this region"
        continue
    fi
    
    while read -r instance_type vcpus; do
        if [ -z "$instance_type" ] || [ -z "$vcpus" ]; then
            continue
        fi
        
        if [ "$vcpus" -gt "$quota_int" ]; then
            echo "  $instance_type: Insufficient quota (needs $vcpus vCPUs, have $quota_int)"
            continue
        fi
        
        availability=$(aws ec2 describe-instance-type-offerings \
            --region "$REGION" \
            --filters "Name=instance-type,Values=$instance_type" \
            --query 'InstanceTypeOfferings[*].InstanceType' \
            --output text 2>/dev/null || echo "")
        
        if [ -n "$availability" ]; then
            gpu_details=$(aws ec2 describe-instance-types \
                --region "$REGION" \
                --instance-types "$instance_type" \
                --query 'InstanceTypes[0].GpuInfo.Gpus[0].[Manufacturer, Name, Count]' \
                --output text 2>/dev/null || echo "")
            
            if [ -n "$gpu_details" ]; then
                gpu_name=$(echo "$gpu_details" | awk '{print $2, $3}')
                gpu_count=$(echo "$gpu_details" | awk '{print $1}')
                GPU_INFO["$instance_type"]="$gpu_count x $gpu_name"
                echo "  ✓ $instance_type: Available ($vcpus vCPUs) - $gpu_count x $gpu_name"
            else
                GPU_INFO["$instance_type"]="GPU info unavailable"
                echo "  ✓ $instance_type: Available ($vcpus vCPUs)"
            fi
            
            AVAILABLE_INSTANCES+=("$instance_type")
        else
            echo "  $instance_type: Not available in region"
        fi
    done <<< "$instance_types"
done

echo ""

if [ ${#AVAILABLE_INSTANCES[@]} -eq 0 ]; then
    echo "No GPU instances available with quota in region $REGION"
    exit 1
fi

echo "=== Available GPU Instances ==="
for i in "${!AVAILABLE_INSTANCES[@]}"; do
    instance="${AVAILABLE_INSTANCES[$i]}"
    gpu="${GPU_INFO[$instance]}"
    echo "$((i+1)). $instance - $gpu"
done
echo ""

read -p "Select instance number to deploy (or 'q' to quit): " selection

if [ "$selection" == "q" ]; then
    echo "Exiting..."
    exit 0
fi

if ! [[ "$selection" =~ ^[0-9]+$ ]] || [ "$selection" -lt 1 ] || [ "$selection" -gt ${#AVAILABLE_INSTANCES[@]} ]; then
    echo "Error: Invalid selection"
    exit 1
fi

SELECTED_INSTANCE="${AVAILABLE_INSTANCES[$((selection-1))]}"
echo ""
echo "Selected: $SELECTED_INSTANCE"
echo ""

echo "=== Setting up resources ==="
echo ""

EXISTING_KEY=$(ls -t gpu-instance-key-*.pem 2>/dev/null | head -n 1)

if [ -n "$EXISTING_KEY" ]; then
    KEY_NAME="${EXISTING_KEY%.pem}"
    echo "Found existing key: $EXISTING_KEY"
    read -p "Use this key? (Y/n): " USE_EXISTING
    
    if [ -z "$USE_EXISTING" ] || [ "$USE_EXISTING" == "Y" ] || [ "$USE_EXISTING" == "y" ]; then
        KEY_FILE="$EXISTING_KEY"
        
        aws ec2 describe-key-pairs \
            --region "$REGION" \
            --key-names "$KEY_NAME" &>/dev/null
        
        if [ $? -eq 0 ]; then
            echo "✓ Using existing key pair: $KEY_NAME"
            echo ""
        else
            echo "⚠ Key pair '$KEY_NAME' not found in AWS region $REGION"
            echo "  The .pem file exists locally but the key isn't registered in AWS."
            echo ""
            read -p "Create new key pair? (Y/n): " CREATE_NEW
            
            if [ -z "$CREATE_NEW" ] || [ "$CREATE_NEW" == "Y" ] || [ "$CREATE_NEW" == "y" ]; then
                TIMESTAMP=$(date +%Y%m%d-%H%M%S)
                KEY_NAME="gpu-instance-key-${TIMESTAMP}"
                KEY_FILE="${KEY_NAME}.pem"
                
                echo "Creating key pair: $KEY_NAME"
                aws ec2 create-key-pair \
                    --region "$REGION" \
                    --key-name "$KEY_NAME" \
                    --query 'KeyMaterial' \
                    --output text > "$KEY_FILE"
                
                chmod 400 "$KEY_FILE"
                echo "✓ Key pair created and saved to: $KEY_FILE"
                echo ""
            else
                echo "Cannot proceed without a valid key pair."
                exit 1
            fi
        fi
    else
        TIMESTAMP=$(date +%Y%m%d-%H%M%S)
        KEY_NAME="gpu-instance-key-${TIMESTAMP}"
        KEY_FILE="${KEY_NAME}.pem"
        
        echo "Creating new key pair: $KEY_NAME"
        aws ec2 create-key-pair \
            --region "$REGION" \
            --key-name "$KEY_NAME" \
            --query 'KeyMaterial' \
            --output text > "$KEY_FILE"
        
        chmod 400 "$KEY_FILE"
        echo "✓ Key pair created and saved to: $KEY_FILE"
        echo ""
    fi
else
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    KEY_NAME="gpu-instance-key-${TIMESTAMP}"
    KEY_FILE="${KEY_NAME}.pem"
    
    echo "No existing key found. Creating new key pair: $KEY_NAME"
    aws ec2 create-key-pair \
        --region "$REGION" \
        --key-name "$KEY_NAME" \
        --query 'KeyMaterial' \
        --output text > "$KEY_FILE"
    
    chmod 400 "$KEY_FILE"
    echo "✓ Key pair created and saved to: $KEY_FILE"
    echo ""
fi

echo "Creating security group..."
SG_TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SG_NAME="gpu-instance-sg-${SG_TIMESTAMP}"
SECURITY_GROUP=$(aws ec2 create-security-group \
    --region "$REGION" \
    --group-name "$SG_NAME" \
    --description "Security group for GPU instance ${SG_TIMESTAMP}" \
    --query 'GroupId' \
    --output text)

echo "✓ Security group created: $SECURITY_GROUP"
echo ""

echo "Adding SSH access rule..."
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
    --region "$REGION" \
    --group-id "$SECURITY_GROUP" \
    --protocol tcp \
    --port 22 \
    --cidr "${MY_IP}/32" > /dev/null

echo "✓ SSH access allowed from your IP: $MY_IP"
echo ""

echo "Getting latest Deep Learning AMI..."

if [[ "$SELECTED_INSTANCE" == p5.* ]]; then
    echo "P5 instance detected, using PyTorch AMI..."
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters "Name=name,Values=Deep Learning OSS Nvidia Driver GPU AMI PyTorch 2.7 (Ubuntu 22.04)*" "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)
    
    if [ -z "$AMI_ID" ] || [ "$AMI_ID" == "None" ]; then
        echo "PyTorch AMI not found, trying base Deep Learning AMI..."
        AMI_ID=$(aws ec2 describe-images \
            --region "$REGION" \
            --owners amazon \
            --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" "Name=state,Values=available" \
            --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
            --output text)
    fi
else
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)
fi

if [ -z "$AMI_ID" ] || [ "$AMI_ID" == "None" ]; then
    echo "Deep Learning AMI not found, using Amazon Linux 2023..."
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters "Name=name,Values=al2023-ami-*-kernel-*-x86_64" "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)
fi

echo "Using AMI: $AMI_ID"
echo ""

echo "Launching instance..."
RESULT=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$SELECTED_INSTANCE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=GPU-Instance-${SG_TIMESTAMP}}]" \
    --output json)

INSTANCE_ID=$(echo "$RESULT" | jq -r '.Instances[0].InstanceId')

echo "✓ Instance launched successfully!"
echo ""

echo "Waiting for instance to get public IP..."
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

echo ""
echo "=========================================="
echo "GPU INSTANCE DEPLOYED SUCCESSFULLY"
echo "=========================================="
echo ""
echo "Instance ID:    $INSTANCE_ID"
echo "Instance Type:  $SELECTED_INSTANCE"
echo "GPU:            ${GPU_INFO[$SELECTED_INSTANCE]}"
echo "Public IP:      $PUBLIC_IP"
echo "Key File:       $KEY_FILE"
echo "Security Group: $SECURITY_GROUP"
echo ""
echo "=========================================="
echo "CONNECTION INSTRUCTIONS"
echo "=========================================="
echo ""

if [ -n "$PUBLIC_IP" ] && [ "$PUBLIC_IP" != "None" ]; then
    echo "Wait ~2 minutes for instance to boot, then connect:"
    echo ""
    echo "  ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
    echo ""
    echo "Or use AWS Systems Manager Session Manager:"
    echo ""
    echo "  aws ssm start-session --target $INSTANCE_ID --region $REGION"
else
    echo "Public IP not yet assigned. Check status:"
    echo ""
    echo "  aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION"
fi

echo ""
echo "=========================================="
echo "INSTANCE MANAGEMENT"
echo "=========================================="
echo ""
echo "Check status:"
echo "  aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""
echo "Stop instance:"
echo "  aws ec2 stop-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""
echo "Terminate instance:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""
echo "=========================================="

