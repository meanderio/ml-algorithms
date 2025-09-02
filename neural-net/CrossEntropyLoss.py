import numpy as np

class CrossEntropyLoss:
    """
    Fused Softmax + Cross-Entropy (numerically stable).
    
    Usage:
        loss = CrossEntropyLoss(label_smoothing=0.0)
        logits = model.forward(X)                 # raw scores (no Softmax layer)
        value  = loss.forward(logits, y)          # y: (N,) int labels or (N,C) one-hot
        dY     = loss.backward()                  # gradient w.r.t. logits (N, C)
    
    Notes:
    - Do NOT add a Softmax layer in your model when using this loss.
    - If you pass one-hot targets, ensure shape matches logits.
    """

    def __init__(self, label_smoothing: float = 0.0, eps: float = 1e-12):
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1).")
        self.label_smoothing = float(label_smoothing)
        self.eps = float(eps)

        # caches
        self._logits  = None
        self._probs   = None
        self._targets = None   # either int labels (N,) or one-hot (N,C)

    def forward(self, logits, targets):
        """
        logits : (N, C) raw scores
        targets: (N,) int labels in [0..C-1] OR (N, C) one-hot
        returns: scalar average loss
        """
        self._logits = np.asarray(logits)
        N, C = self._logits.shape

        # Stable softmax
        x           = self._logits - np.max(self._logits, axis=1, keepdims=True)
        exps        = np.exp(x)
        self._probs = exps / np.sum(exps, axis=1, keepdims=True)

        # Prepare targets as one-hot (with optional label smoothing)
        if targets.ndim == 1:
            y_idx = targets.astype(int)
            if np.any((y_idx < 0) | (y_idx >= C)):
                raise ValueError("Class indices out of range.")
            y_oh = np.zeros((N, C), dtype=self._logits.dtype)
            y_oh[np.arange(N), y_idx] = 1.0
        elif targets.ndim == 2:
            if targets.shape != (N, C):
                raise ValueError("One-hot targets must have shape (N, C).")
            y_oh = targets.astype(self._logits.dtype)
        else:
            raise ValueError("targets must be (N,) int labels or (N, C) one-hot.")

        if self.label_smoothing > 0.0:
            ls = self.label_smoothing
            y_oh = (1 - ls) * y_oh + ls / C

        self._targets = y_oh  # cache as one-hot-like

        # Cross-entropy: -sum(y * log(p)) averaged over batch
        # clip to avoid log(0)
        p = np.clip(self._probs, self.eps, 1.0)
        loss = - np.mean(np.sum(self._targets * np.log(p), axis=1))
        return (1/N) * float(loss)

    def backward(self):
        """
        Returns dL/dlogits of shape (N, C).
        For CE with softmax: dL/dz = (p - y) / N
        """
        if self._probs is None or self._targets is None:
            raise RuntimeError("Call forward() before backward().")
        N = self._probs.shape[0]
        return (self._probs - self._targets) / N
    
    def step(self,):
        pass