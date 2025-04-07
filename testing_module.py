from classifier import *
model=MammogramCNN()
model.load_state_dict(torch.load("mammogram_cnn.pth"))
model.eval()
model.to("cpu")

test_dataset = MammogramDataset(
    tensor_folder="tensor_dataset/",  # adjust path if needed
    label_file="modified_labels.csv",
    transform=transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
