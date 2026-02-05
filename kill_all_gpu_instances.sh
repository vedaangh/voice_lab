#!/bin/bash

set -e

REGION="${1:-$(aws configure get region 2>/dev/null || echo 'us-east-1')}"

echo "=== Killing all AWS GPU instances ==="
echo "Region: $REGION"
echo ""

INSTANCE_IDS=$(aws ec2 describe-instances \
    --region "$REGION" \
    --filters "Name=instance-state-name,Values=running,pending,stopped" \
    --query 'Reservations[*].Instances[*].[InstanceId,InstanceType]' \
    --output text | grep -E '(g4dn|g5|g6|p3|p4d|p4de|p5)' | awk '{print $1}')

if [ -z "$INSTANCE_IDS" ]; then
    echo "No GPU instances found in region $REGION"
    exit 0
fi

echo "Found GPU instances to terminate:"
aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids $INSTANCE_IDS \
    --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,State.Name]' \
    --output table

echo ""
read -p "Are you sure you want to TERMINATE these instances? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

echo "$INSTANCE_IDS" | xargs aws ec2 terminate-instances --region "$REGION" --instance-ids

echo ""
echo "Termination initiated for all GPU instances."















