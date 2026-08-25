"""Single-instance benchmark rig: dedicated VPC, no inbound, SSM for access.

Local NVMe is the point of the exercise, so the instance type must carry
instance storage -- EBS would measure the network, not the format.
"""

import pulumi
import pulumi_aws as aws

cfg = pulumi.Config()
INSTANCE_TYPE = cfg.get("instanceType") or "i4i.2xlarge"

tags = {"Project": "polars-pylance-bench", "ManagedBy": "pulumi"}

# -- network ---------------------------------------------------------------
vpc = aws.ec2.Vpc(
    "bench-vpc",
    cidr_block="10.42.0.0/16",
    enable_dns_hostnames=True,
    enable_dns_support=True,
    tags={**tags, "Name": "bench-vpc"},
)
igw = aws.ec2.InternetGateway("bench-igw", vpc_id=vpc.id, tags=tags)
azs = aws.get_availability_zones(state="available")
subnet = aws.ec2.Subnet(
    "bench-subnet",
    vpc_id=vpc.id,
    cidr_block="10.42.1.0/24",
    availability_zone=azs.names[0],
    map_public_ip_on_launch=True,
    tags={**tags, "Name": "bench-subnet"},
)
rt = aws.ec2.RouteTable(
    "bench-rt",
    vpc_id=vpc.id,
    routes=[aws.ec2.RouteTableRouteArgs(cidr_block="0.0.0.0/0", gateway_id=igw.id)],
    tags=tags,
)
aws.ec2.RouteTableAssociation("bench-rta", subnet_id=subnet.id, route_table_id=rt.id)

# Egress only: access is via SSM, so nothing needs to reach the box inbound.
sg = aws.ec2.SecurityGroup(
    "bench-sg",
    vpc_id=vpc.id,
    description="egress only; access via SSM Session Manager",
    egress=[
        aws.ec2.SecurityGroupEgressArgs(
            protocol="-1", from_port=0, to_port=0, cidr_blocks=["0.0.0.0/0"]
        )
    ],
    tags=tags,
)

# -- instance role (SSM only) ----------------------------------------------
role = aws.iam.Role(
    "bench-role",
    assume_role_policy="""{
      "Version": "2012-10-17",
      "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"},
                     "Action": "sts:AssumeRole"}]
    }""",
    tags=tags,
)
aws.iam.RolePolicyAttachment(
    "bench-ssm",
    role=role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
)
profile = aws.iam.InstanceProfile("bench-profile", role=role.name, tags=tags)

AMI_PARAM = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
ami = aws.ssm.get_parameter(name=AMI_PARAM)

# Mount the instance-store NVMe and install uv; everything else happens over SSM.
user_data = r"""#!/bin/bash
set -eux
exec > >(tee /var/log/bench-init.log) 2>&1
dnf install -y nvme-cli xfsprogs git

DEV=""
for d in /dev/nvme*n1; do
  nvme id-ctrl "$d" 2>/dev/null | grep -qi "Instance Storage" || continue
  DEV="$d"; break
done
if [ -z "$DEV" ]; then echo "FATAL: no instance-store NVMe found"; lsblk; exit 1; fi

mkfs.xfs -f "$DEV"
mkdir -p /mnt/nvme
mount -o noatime "$DEV" /mnt/nvme
chmod 1777 /mnt/nvme
df -h /mnt/nvme

export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
install -m 0755 /root/.local/bin/uv /usr/local/bin/uv || true

touch /mnt/nvme/READY
echo "init complete"
"""

instance = aws.ec2.Instance(
    "bench-instance",
    instance_type=INSTANCE_TYPE,
    ami=ami.value,
    subnet_id=subnet.id,
    vpc_security_group_ids=[sg.id],
    iam_instance_profile=profile.name,
    user_data=user_data,
    root_block_device=aws.ec2.InstanceRootBlockDeviceArgs(
        volume_size=40, volume_type="gp3", delete_on_termination=True
    ),
    tags={**tags, "Name": "polars-pylance-bench"},
)

pulumi.export("instance_id", instance.id)
pulumi.export("instance_type", instance.instance_type)
pulumi.export("az", instance.availability_zone)
