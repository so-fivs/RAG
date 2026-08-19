data "aws_caller_identity" "current" {}

locals {
  bucket_name = var.bucket_name != "" ? var.bucket_name : "${var.project_name}-pdfs-${data.aws_caller_identity.current.account_id}"
  pdf_files   = fileset(var.pdfs_local_path, "*.pdf")
}

# ---------------------------------------------------------------------------
# S3 — bucket público de solo lectura con los PDFs de FDS
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "fds_pdfs" {
  bucket = local.bucket_name

  tags = {
    Project = var.project_name
  }
}

resource "aws_s3_bucket_public_access_block" "fds_pdfs" {
  bucket = aws_s3_bucket.fds_pdfs.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_policy" "public_read" {
  bucket = aws_s3_bucket.fds_pdfs.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "PublicReadGetObject"
      Effect    = "Allow"
      Principal = "*"
      Action    = "s3:GetObject"
      Resource  = "${aws_s3_bucket.fds_pdfs.arn}/fds/*"
    }]
  })

  depends_on = [aws_s3_bucket_public_access_block.fds_pdfs]
}

resource "aws_s3_object" "pdfs" {
  for_each = local.pdf_files

  bucket       = aws_s3_bucket.fds_pdfs.id
  key          = "fds/${each.value}"
  source       = "${var.pdfs_local_path}/${each.value}"
  etag         = filemd5("${var.pdfs_local_path}/${each.value}")
  content_type = "application/pdf"
}

# ---------------------------------------------------------------------------
# EC2 — backend FastAPI + ChromaDB + Ollama
# ---------------------------------------------------------------------------
data "aws_iam_instance_profile" "lab" {
  name = var.lab_instance_profile_name
}

data "aws_vpc" "default" {
  default = true
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_security_group" "backend" {
  name        = "${var.project_name}-backend-sg"
  description = "SSH + API RAG FDS"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_allowed_cidr]
  }

  ingress {
    description = "API FastAPI"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.api_allowed_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project = var.project_name
  }
}

resource "aws_instance" "backend" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.backend.id]
  iam_instance_profile   = data.aws_iam_instance_profile.lab.name

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/templates/deploy.sh.tpl", {
    repo_url    = var.repo_url
    repo_branch = var.repo_branch
    bucket_name = aws_s3_bucket.fds_pdfs.id
  })

  tags = {
    Name    = "${var.project_name}-backend"
    Project = var.project_name
  }

  depends_on = [aws_s3_object.pdfs]
}
