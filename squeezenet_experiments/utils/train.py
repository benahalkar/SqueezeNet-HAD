import torch
import torch.optim as optim
import time

noise_std = 0.1

def train_model(model, device, optimizer, loss_criteria, train_dataloader, val_dataloader, epochs, restarts, save_path, add_noise=False):
    train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history, epoch_times = [], [], [], [], []

    higest_val_acc = None
    
    for restart in range(restarts):
        # defining the training scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # -----------Training-----------
        for epoch in range(epochs):

            running_loss = 0.0
            iteration_value = 0
            correct_predictions = 0
            total_predictions = 0

            model.train(True)

            start = time.time()

            for i, data in enumerate(train_dataloader):

                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                output = model(images)

                loss = loss_criteria(output, labels)

                loss.backward()
                optimizer.step()

                if add_noise: 
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad += torch.randn_like(param.grad) * noise_std


                running_loss = running_loss + loss.item()
                iteration_value = iteration_value + 1

                output = torch.argmax(output, dim=1)
                correct_predictions = correct_predictions + torch.sum(labels == output)
                total_predictions = total_predictions + len(labels)

            training_loss = round(float(running_loss)/iteration_value, 8)
            train_loss_history.append(training_loss)

            training_acc = round(float(correct_predictions)*100/total_predictions, 2)
            train_accuracy_history.append(training_acc)

            running_loss = 0.0
            iteration_value = 0
            correct_predictions = 0
            total_predictions = 0

            model.eval()

            # -----------Validation Phase-----------
            with torch.no_grad():
                for i, data in enumerate(val_dataloader):

                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)

                    output = model(images)
                    loss = loss_criteria(output, labels)

                    running_loss = running_loss + loss.item()
                    iteration_value = iteration_value + 1

                    output = torch.argmax(output, dim=1)
                    correct_predictions = correct_predictions + torch.sum(labels == output)
                    total_predictions = total_predictions + len(labels)

                validation_loss = round(float(running_loss)/iteration_value, 8)
                val_loss_history.append(validation_loss)

                validation_acc = round(float(correct_predictions)*100/total_predictions, 2)
                val_accuracy_history.append(validation_acc)

            # -----------Progress update-----------
            if higest_val_acc is None or validation_acc > higest_val_acc:
                higest_val_acc = validation_acc
                torch.save(model.state_dict(), save_path)

            time_delta = round(time.time() - start, 2)
            epoch_times.append(time_delta)
            time_delta = "{:.2f}".format(time_delta)

            print(f"Time delta {time_delta} s. Epoch {epoch + 1}/{epochs} --- Train loss {training_loss} --- Train acc {training_acc} --- Val loss {validation_loss} --- Val acc {validation_acc}")

    return train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history, epoch_times


def do_inference(model, images):
    model.eval()
    with torch.no_grad():
        output = model(images)
        output = torch.argmax(output, dim=1)
        
    return output

def do_transpose(images):
    return torch.transpose(images, 0, 2)
