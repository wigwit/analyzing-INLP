""" Module to define a linear classifier """

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import random


## defining GPU here
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

class LinearClassifier(torch.nn.Module):
    def __init__(self,
                 input_embeddings: torch.tensor,
                 output: torch.tensor,
                 tag_size: int,
                 device: torch.device,
                 random_seed: int = None):
        """
        Defines a categorical linear classifier

        Params:
        -------
        input_embeddings
            A tensor with individual embeddings along the horizontal axis.
        output
            A tensor of the appropriate target label for each input embedding.
        tag_size
            The number of categories for the linear classifier to predict.
        device
            Specify which device (cpu vs gpu) the linear classifier should be trained on.
        random_seed
            Optionally specify the random seed, for reproducability.
        """

		# Inherit attributes and methods from pytorch Neural Network
        super().__init__()

        # Set the random seed, if provided
        if random_seed:
            random.seed(random_seed)

        # Specify the device to be used
        self.device = device

        # Define training data
        self.embeddings = input_embeddings
        self.output = output
        self.linear = torch.nn.Linear(input_embeddings.shape[1], tag_size, device=device, dtype=torch.double)

        # Define loss function for training
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.tensor) -> torch.tensor:
        """
        Performs a forward pass through the linear classifier

        Params:
        -------
        embeddings
            A tensor with individual embeddings along the horizontal axis.

        Returns:
        --------
        forward_res
            The tensor that results from passing the embeddings through the linear classifier.
        """
        # Format embeddings
        input_embeddings = embeddings.to(device)
        input_embeddings = input_embeddings.double()

        # Pass embeddings through the classifier
        forward_res = self.linear(input_embeddings)

        return forward_res

# TODO: confirm this output is a float, and what the number represents
    def eval(self,
             input_embeddings: torch.tensor,
             output_categories: torch.tensor) -> tuple(torch.tensor, float):
        """
        Evaluate model performance on external (non-training) data

        Params:
        -------
        input_embeddings
            The input tensor to be passed through the linear classifier
        output_categories
            The actual output categories, to be assessed against the linear classifier estimates

        Returns:
        --------
        model_predictions
            The model's predicted distribution
        loss
            The error????


        """
        # Run the input embeddings through the model
        with torch.no_grad():
            input_embeddings = input_embeddings.to(device)
            output_categories = output_categories.to(device)
            model_predictions = self.forward(input_embeddings)
            loss = self.loss_func(model_predictions, output_categories)

        # Report the accuracy of the most likely category predicted by the model for each input
        model_categories = torch.argmax(model_predictions, dim=1).cpu().numpy()
        print(f'dev accuracy score:{accuracy_score(output_categories.cpu().numpy(), model_categories):.4f}')

        return model_predictions, loss.item()

# TODO: what datatype is *args???
    def batched_input(self,
                      *args,
                      batch_size: int = 64) -> DataLoader:
        """
        Description

        Params:
        -------
        *args
            DESCRIPTION NEEDED
        batch_size
            The number of embeddings to include in each batch.

        Returns:
        --------
        dataloader
            A DataLoader object that is configured to batch the data to the specified batch size.
        """
        data_set = TensorDataset(args[0], args[1])
        dataloader = DataLoader(data_set, batch_size=batch_size)

        return dataloader

    def optimize(self,
                 lr: float = 0.001,
                 num_epochs: int = 500) -> tuple(torch.nn.Linear, float):
        """
        Fits the linear classifier using the training data provided.

        Parameters:
        -----------
        lr
            Learning rate used during training.
        num_epochs
            The number of epochs to run when training.

        Returns:
        --------
        best_model
            The best-fit linear model seen during training.
        train_accuracy
            The accuracy achieved by the best-fit linear model over the training data set.
        """

        # Initialize model training parameters
        optimizer = torch.optim.AdamW(self.linear.parameters(), lr=lr)
        best_predictions = None
        best_loss = float('inf')
        stop_count = 0
        output = self.output.to(device)
        dataloader = self.batched_input(self.embeddings, output)

        # Run training
        for epoch in range(num_epochs):
            predictions = []
            total_loss = 0
            for embedding, label in dataloader:
                optimizer.zero_grad()
                prediction_i = self.forward(embedding)
                loss = self.loss_func(prediction_i, label)
                loss.backward(retain_graph=True)
                optimizer.step()
                prediction_i = prediction_i.to('cpu')
                predictions.append(prediction_i)
                total_loss += loss.item()

            total_loss = total_loss / len(dataloader)
            predictions = torch.cat(predictions)

            # Implement stopping criterion
            if total_loss < best_loss:
                best_loss = total_loss
                best_model = self.linear
                best_predictions = predictions
                stop_count = 0
            else:
                if stop_count == 5:
                    break
                else:
                    stop_count += 1

        # Get accuracy for the best predicted categories for the input embeddings
        best_category_prediction = torch.argmax(best_predictions, dim=1).cpu().numpy()
        train_accuracy = accuracy_score(self.output.numpy(), best_category_prediction)
        print(f'train accuracy score:{train_accuracy:.4f}')

        return best_model, train_accuracy





