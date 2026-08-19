output "bucket_name" {
  value = aws_s3_bucket.fds_pdfs.id
}

output "pdfs_public_url_example" {
  description = "Ejemplo de URL pública de un PDF del bucket"
  value       = length(local.pdf_files) > 0 ? "https://${aws_s3_bucket.fds_pdfs.bucket_regional_domain_name}/fds/${tolist(local.pdf_files)[0]}" : "sin PDFs en ${var.pdfs_local_path}"
}

output "ec2_public_ip" {
  value = aws_instance.backend.public_ip
}

output "api_url" {
  value = "http://${aws_instance.backend.public_ip}:8000"
}

output "ssh_command" {
  value = "ssh -i <ruta-a-tu-key>.pem ubuntu@${aws_instance.backend.public_ip}"
}
