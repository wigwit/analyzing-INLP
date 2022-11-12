""" Module for running the INLP loop to guard protected linear information"""

# Libraries
import numpy as np
import scipy
import torch

from numpy.linalg import matrix_rank

from linear_classifier import LinearClassifier

def get_rowspace_projection(linear_matrix: np.array) -> np.array:
    """
    Defines the rowspace projection onto the vectorspace spanned by the columns of a 2D matrix.

    Parameters:
    -----------
    linear_matrix
        The 2D matrix used to perform classification in a linear classifier.

    Returns:
    --------
    rowspace_projection_matrix
        The 2D matrix that, when multiplied by an embedding, results in the projection of that embedding onto the
        vectorspace spanned by the columns of linear_matrix.
    """

    # Get the orthogonal basis of the rowspace
    if np.allclose(linear_matrix, 0):
        linear_matrix_basis = np.zeros_like(linear_matrix.T)
    else:
        linear_matrix_basis = scipy.linalg.orth(linear_matrix.T)

    # Handle sign ambiguity
    linear_matrix_basis = linear_matrix_basis * np.sign(linear_matrix_basis[0][0])

    # Get the projection matrix onto the rowspace
    rowspace_projection_matrix = linear_matrix_basis.dot(linear_matrix_basis.T)

    return rowspace_projection_matrix


def get_projection_to_intersection_of_nullspaces(rowspace_projection_matrices: list[np.ndarray]) -> np.array:
    """
    Determines the matrix that projects onto the intersection of the nullspaces of the provided
    rowspace_projection_matrices.

    Details:
    --------
    Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n), this function calculates the projection to
    the intersection of all nullspasces of the matrices w_1, ..., w_n. It uses the intersection-projection formula of
    Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
    N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))

    Parameters:
    -----------
    rowspace_projection_matrices
        A list of matrices. Each matrix projects embeddings onto the rowspace of its respective linear classifier.

    Returns:
    --------
    nullspace_projection_matrix
        The 2D matrix that, when multiplied by an embedding, results in the projection of that embedding onto the
        intersection of the nullspaces of the provided rowspace projection matrices.
    """

    # Get the union of the rowspace projection matrices
    input_dim = rowspace_projection_matrices[0].shape[0]
    I = np.eye(input_dim)
    rowspace_union = np.sum(rowspace_projection_matrices, axis=0)

    # Get the intersection of the nullspaces of the rowspace projection matrices
    nullspace_projection_matrix = I - get_rowspace_projection(rowspace_union)

    return nullspace_projection_matrix


def reinitialize_classifier(in_size: int,
                            out_size: int,
                            device: torch.device,
                            random_seed: int = None) -> torch.nn.Linear:
    """
    Reinitializes the linear classifier to be used in a new training round of INLP.

    Parameters:
    -----------
    in_size
        The dimension of the inputs to the linear model.
    out_size
        The dimension of the outputs to the linear model. Should be equivalent to the number of target classes.
    device
        Specify which device (cpu vs gpu) the linear classifier should be trained on.
    random_seed
        Optionally specify the random seed, for reproducability.

    Returns:
    --------
    linear_model
        A newly-initialized, untrained linear model.
    """

    # Specify the random seed, if applicable
    if random_seed:
        random.seed(random_seed)

    # Initialize a blank linear model
    linear_model = torch.nn.Linear(in_size, out_size, device=device, dtype=torch.double)

    return linear_model


def apply_projection(original_embeddings: torch.tensor,
                     projection_matrix: np.array) -> torch.tensor:
    '''
    Projects the original_embeddings onto the subspace spanned by the projection_matrix.

    Parameters:
    -----------
    original_embeddings
        A 2D matrix for which each row is an embedding to be projected onto the space specified by the
        projection_matrix.
    projection_matrix
        A 2D matrix that, when multiplied by an embedding, results in the projection of that embedding onto the
        vectorspace spanned by the matrix's columns.

    Returns:
    --------
    projected_embeddings
        The embeddings that result from applying the projection. They preserve the order of the original embeddings.
    '''

    # Cast the projection matrix to a torch tensor
    projection_matrix = torch.tensor(projection_matrix, dtype=torch.double)

    # Get the projected embeddings
    projected_embeddings = torch.matmul(projection_matrix, original_embeddings.T)
    projected_embeddings = projected_embeddings.T.double()

    return projected_embeddings

def perform_inlp_loop(original_embeddings: torch.tensor,
                      target_output: torch.tensor,
                      num_classes: int,
                      min_accuracy: float,
                      verbose: bool = False) -> tuple(torch.tensor, torch.tensor):
    """
    Performs the INLP loop to guard the linear information in the original_embeddings that is used to predict the
    classes provided in the target_output.

    Details:
    --------
    This approach works by fitting a linear classifier to the data that predicts the protected classes, then projecting
    the embeddings onto the nullspace of that classifier, effectively removing from the embeddings the information that
    was used by the linear classifier to perform its classification. This process is then repeated with the projected
    embeddings, since there may still be information present in the projections that a different linear classifier could
    use to determine the protected classes. We repeat this process until a linear classifier trained on the resulting
    projections is unable to outperform the specified min_accuracy.

    Paramters:
    ----------
    original_embeddings
        A 2D matrix for which each row is an embedding to be projected onto the space specified by the
        projection_matrix.
    target_output
        words
    num_classes
        The number of classes that the linear classifier has to select between when making predictions.
    min_accuracy
        The minimum accuracy that a linear classifier must attain in order for us to conclude that the inputs it was
        trained on contain linear information about the protected classes that the linear classifier is trying to
        predict. Typically, this is equivalent to the accuracy that a linear classifier would attain if it were to guess
        the most common class in all cases.
    verbose
        A flag that indicates whether the inlp loop should print the linear classifier accuracy and projection matrix
        dimensions to standard output.

    Returns:
    --------
    all_nullspace_projection_matrix
        The 2D projection matrix which, when multipled by an embeddings, results in a projection of that embedding onto
        the intersection of the nullspaces of each of the linear classifiers fit in the inlp loop.
    current_embeddings
        A 2D matrix for which each row is a projection of the corresponding row from original_embeddings onto the
        intersection of the nullspaces of each of the linear classifiers fit in the inlp loop. These projections are
        said to be linearly guarded against the information used to predict the protected classes in target_output.
    """

    # Initialize matrices for inlp
    I = np.eye(original_embeddings.shape[1])
    all_nullspace_projection_matrix = I
    linear_classifier_weights = []
    nullspace_projection_matrices = []
    rowspace_projections = []
    current_embeddings = original_embeddings

    # Initialize thresholds for inlp
    if verbose:
        i = 0
    current_accuracy = 100 * min_accuracy

    # INLP loop
    while current_accuracy > min_accuracy:

        # Train new linear model
        linear_model = LinearClassifier(input_embeddings=current_embeddings,
                                        output=target_output,
                                        tag_size=num_classes)
        fit_model, current_accuracy = linear_model.optimize()
        model_weights = fit_model.weight.detach().cpu().numpy()
        linear_classifier_weights.append(model_weights)

        # Get the rowspace projection matrix for the newly trained model
        rowspace_projection_matrix = get_rowspace_projection(model_weights)
        rowspace_projections.append(rowspace_projection_matrix)

        # Get the nullspace projection matrix for the newly trained model
        current_nullspace_projection_matrix = get_projection_to_intersection_of_nullspaces(
            input_dim=rowspace_projection_matrix.shape[0],
            rowspace_projection_matrices=rowspace_projections
        )

        # Get the matrix that projects onto the intersection of the linear classifiers nullspaces
        all_nullspace_projection_matrix = np.matmul(current_nullspace_projection_matrix,
                                                    all_nullspace_projection_matrix)
        nullspace_projection_matrices.append(all_nullspace_projection_matrix)
        current_embeddings = apply_projection(original_embeddings, all_nullspace_projection_matrix)

        # Reset the linear model parameter cache
        linear_model.linear.reset_parameters()

        # Print updated results to standard output, if applicable
        if verbose:
            projection_rank = matrix_rank(all_nullspace_projection_matrix)
            embedding_rank = matrix_rank(current_embeddings)
            i += 1
            print(f'Iteartion {i}.\n '+
                  f'Accuracy is {current_accuracy} \n'+
                  f'Projection matrix rank is {projection_rank}\n'+
                  f'Embedding rank is {embedding_rank}\n')

    return all_nullspace_projection_matrix, current_embeddings
