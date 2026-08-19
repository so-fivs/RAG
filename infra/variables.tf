variable "aws_region" {
  description = "Región del Learner Lab (normalmente us-east-1)"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Prefijo para nombrar los recursos"
  type        = string
  default     = "rag-fds"
}

variable "bucket_name" {
  description = "Nombre del bucket S3 (debe ser único global). Si no se define, se genera uno con random suffix."
  type        = string
  default     = ""
}

variable "pdfs_local_path" {
  description = "Ruta local a data/bronze con los PDFs de FDS a subir al bucket"
  type        = string
  default     = "../data/bronze"
}

# --- EC2 ---
variable "instance_type" {
  description = "Tipo de instancia EC2. Ollama con qwen2.5:1.5b + llava:7b necesita >= 4 vCPU/16GB (t3.xlarge). Verifica qué tipos permite tu Learner Lab."
  type        = string
  default     = "t3.xlarge"
}

variable "key_name" {
  description = "Nombre del key pair EC2 (vockey en AWS Academy) para SSH"
  type        = string
}

variable "ssh_allowed_cidr" {
  description = "CIDR permitido para SSH (pon tu IP pública /32, nunca 0.0.0.0/0)"
  type        = string
}

variable "api_allowed_cidr" {
  description = "CIDR permitido para acceder al API (8000). 0.0.0.0/0 si quieres que sea público."
  type        = string
  default     = "0.0.0.0/0"
}

variable "lab_instance_profile_name" {
  description = "Nombre del instance profile IAM ya existente en el lab (AWS Academy usa 'LabInstanceProfile'). No se puede crear uno nuevo en Learner Lab."
  type        = string
  default     = "LabInstanceProfile"
}

variable "repo_url" {
  description = "URL del repositorio git a clonar en la instancia"
  type        = string
  default     = "https://github.com/so-fivs/RAG.git"
}

variable "repo_branch" {
  description = "Branch a desplegar"
  type        = string
  default     = "main"
}
