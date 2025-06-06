import torch

print("--- Iniciando verificação de GPU com PyTorch ---")

if torch.cuda.is_available():
    print("Sucesso! PyTorch encontrou uma GPU compatível com CUDA.")
    gpu_count = torch.cuda.device_count()
    print(f"Número de GPUs encontradas: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("Falha! Nenhuma GPU compatível com CUDA foi encontrada pelo PyTorch.")

print("--- Verificação finalizada ---")