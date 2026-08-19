terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# AWS Academy Learner Lab: las credenciales son temporales (aws_access_key_id,
# aws_secret_access_key, aws_session_token) y se toman de ~/.aws/credentials
# (perfil "default") o de las variables de entorno AWS_*. Si corres esto desde
# Cloud9 dentro del lab, el provider usa el rol de la instancia automáticamente
# y no necesitas configurar nada.
provider "aws" {
  region = var.aws_region
}
