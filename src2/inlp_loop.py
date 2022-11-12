""" Module for running the INLP loop to guard protected linear information"""

class INLPTraining(LinearClassifier):

    def __init__(self, input_embeddings, output, tag_size):
        super().__init__(input_embeddings, output, tag_size)
        self.input_dim = self.embeddings.shape[1]
        # only used for the applying projection
        self.original_embedding = input_embeddings.T

    def get_rowspace_projection(self, model_weight):
        W = model_weight
        if np.allclose(W, 0):
            w_basis = np.zeros_like(W.T)
        else:
            w_basis = scipy.linalg.orth(W.T)  # orthogonal basis
        w_basis = w_basis * np.sign(w_basis[0][0])  # handle sign ambiguity
        P_W = w_basis.dot(w_basis.T)  # orthogonal projection on W's rowspace
        return P_W

    def get_projection_to_intersection_of_nullspaces(self, input_dim, rowspace_projection_matrices: List[np.ndarray]):
        """
        Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
        this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
        uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
        N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
        :param rowspace_projection_matrices: List[np.array], a list of rowspace projections
        :param input_dim: input dim
        """
        # This is werid because Q is not normalized so the N(P) = I-P does not work
        I = np.eye(input_dim)
        Q = np.sum(rowspace_projection_matrices, axis=0)
        P = I - self.get_rowspace_projection(Q)
        return P

    def reinitialize_classifier(self):
        ## may be empty cache here
        in_size = self.linear.in_features
        out_size = self.linear.out_features
        random.seed(42)
        self.linear = torch.nn.Linear(in_size, out_size, device=device, dtype=torch.double)

    def apply_projection(self, P):
        '''
        applying projection of P to the embedding vectors
        '''
        ## may be empty cache here
        P = torch.tensor(P, dtype=torch.float)
        self.embeddings = torch.matmul(P, self.original_embedding).T
        self.embeddings = self.embeddings.double()

    def run_INLP_loop(self, iteration, dev_x=None, dev_y=None, min_acc=0.0):
        I = np.eye(self.input_dim)
        P = I
        Ws = []
        all_P = []
        rowspace_projections = []
        for i in range(iteration):
            self.reinitialize_classifier()
            bm, acc = self.optimize()
            if dev_x is not None:
                dum1, dum2, acc = self.eval(dev_x, dev_y)
                print(f'dev acc for round {i} is {acc:.4f}')
            if acc < min_acc:
                # TODO: not sure it should be continue here
                break
            W = bm.weight.detach().cpu().numpy()
            Ws.append(W)
            # Noted this is the projection space for W, not the null space
            P_rowspace_wi = self.get_rowspace_projection(W)
            rowspace_projections.append(P_rowspace_wi)
            # This line is supposed to get the null space for the projection space of W
            # Intuitively I think the rank makes sense, but I don't know how this will hold
            P_Nwi = self.get_projection_to_intersection_of_nullspaces(input_dim=P_rowspace_wi.shape[0],
                                                                      rowspace_projection_matrices=rowspace_projections)
            # This line is what they showed originally but the function looks weird
            # P = self.get_projection_to_intersection_of_nullspaces(rowspace_projections)
            P = np.matmul(P_Nwi, P)
            all_P.append(P)
            self.apply_projection(P)

        return P, rowspace_projections, Ws, all_P