# Create DataLoader
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Test DataLoader
for batch_images, batch_labels in dataloader:
    print("Batch Shape:", batch_images.shape)  # Should be (batch_size, 1, 1024, 1024)
    print("Labels:", batch_labels)
    break
