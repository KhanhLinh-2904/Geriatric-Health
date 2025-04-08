import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from CNN1D import CNN1D  

folder_path = "splits"
save_fold = "best_model"
files = os.listdir(folder_path)

for data in files:
    data_path = os.path.join(folder_path, data)
    # Load Data
    loaded_data = np.load(data_path)
    x_train = loaded_data["x_train"]
    y_train = loaded_data["y_train"]
    x_val = loaded_data["x_val"]
    y_val = loaded_data["y_val"]

    # Convert to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    # Create DataLoader for training 
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Create DataLoader for validation
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Training Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(input_size=x_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    num_epochs = 1000
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    best_epoch = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0.0
        with torch.no_grad():  # Disable gradient computation during validation
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Print training and validation loss
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Save the best model based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict()

    # Save the best model
    name_best_model = data + "_best_model.pth"
    name_best_model_path = os.path.join(save_fold, name_best_model)
    torch.save(best_model_state, name_best_model_path)
    print(f"Best Model Saved at Epoch {best_epoch} with Validation Loss: {best_loss:.4f}")
    loss_curve = os.path.join("loss_curve_figure", data+"_loss_curve.png")
    # Plot Loss Curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(loss_curve)
    plt.show()
