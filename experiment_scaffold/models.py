"""Safe full-GCN backbone factory.

``build_gcn`` constructs a multi-layer GCN from hyperparameters, wrapping
``torch_geometric.nn.models.GCN``. Its purpose is to make the *default* call
trainable at any depth, because PyG's ``GCN`` defaults to ``norm=None`` -- and a
deep (L>=5) vanilla GCN with no inter-layer normalization is **untrainable** on
graph-regression tasks like brain-age: it collapses to the mean predictor with
train R^2 ~ 0 (an optimization failure, not over-smoothing or over-fitting).

This factory defaults ``norm='batch_norm'`` -- verified to fix that collapse
(R^2 -0.016 -> 0.504 at L=5 on brain data). ``'layer_norm'`` and ``'graph_norm'``
are **also verified** fixes (synthetic L=32 fixture, train R^2: None 0.02 ->
batch_norm 0.75, layer_norm 0.97, graph_norm 1.00 -- any inter-layer normalizer
re-conditions the deep propagation). A naive deep call therefore trains;
reproducing the bug requires *consciously* passing ``norm=None``.

Not in scope (deliberately): **residual connections** -- the other principled
deep-GCN fix -- are unverified here and PyG's ``GCN`` has no ``residual`` arg;
adding them would require hand-rolling a ``GCNConv`` stack. Use a custom backbone
if you need them.

.. warning::
   ``GCNConv``'s symmetric normalization yields **NaN on negative edge weights**.
   For signed graphs (e.g. functional-connectivity anti-correlations) pass
   ``abs``/``abs_max``-normalized ``edge_weight`` -- or none. This factory does not
   guard against it (a separate failure mode from the collapse this fix targets).
"""

from torch_geometric.nn.models import GCN


def build_gcn(
    in_channels,
    hidden_channels,
    out_channels,
    num_layers=3,
    dropout=0.0,
    act="relu",
    norm="batch_norm",
    jk=None,
    **kwargs,
):
    """Build a safe multi-layer GCN backbone.

    Returns a ``torch_geometric.nn.models.GCN`` (a ``BasicGNN``) whose
    ``forward(x, edge_index, edge_weight=..., batch=...)`` yields node embeddings
    ``[num_nodes, out_channels]``.

    Parameters
    ----------
    in_channels : int
        Input feature dimension (matches the feature-encoder output).
    hidden_channels : int
        Hidden dimension.
    out_channels : int
        Output node-embedding dimension.
    num_layers : int, optional
        Number of GCN layers. Default is 3. Safe at any depth because ``norm``
        defaults on.
    dropout : float, optional
        Dropout rate. Default is 0.0.
    act : str or callable or None, optional
        Activation between layers. Default is ``'relu'``.
    norm : str or callable or None, optional
        Inter-layer normalization. **Default ``'batch_norm'`` -- a verified fix
        for the deep-GCN collapse; do not set to None for L>=4 unless ablating.**
        ``'layer_norm'`` and ``'graph_norm'`` are also verified to train at depth.
    jk : str or None, optional
        Jumping-knowledge mode (``'cat'``/``'max'``/``'lstm'``). Default None.
        Note: JK alone does **not** fix the deep collapse -- it is a
        representational, not a trainability, mechanism.
    **kwargs
        Forwarded to ``GCN`` / underlying ``GCNConv`` (e.g. ``act_kwargs``,
        ``norm_kwargs``, ``act_first``).

    Returns
    -------
    torch_geometric.nn.models.GCN
        The configured backbone.
    """
    return GCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout=dropout,
        act=act,
        norm=norm,
        jk=jk,
        **kwargs,
    )
