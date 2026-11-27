"""覆盖联合图的构造、填充和单样本训练。可用 unittest discover -s tests 运行。"""

import unittest

import torch

from molbridge.data import collate_pairs, molecule_graph
from molbridge.model import MolBridge
from molbridge.train import batch_loss, evaluate


class MolBridgeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.model = MolBridge(3, hidden_dim=8, ffn_dim=16, num_heads=2,
                               num_layers=2, dropout=0.0, classifier_dim=8)
        self.pair = (molecule_graph("CO"), molecule_graph("N"), 2)

    def test_chemical_bonds_and_full_molecule(self):
        """一个含 60 个原子的分子应完整保留，药物之间的化学键块应为零。"""
        long_pair = (molecule_graph("C" * 60), molecule_graph("O"), 0)
        features, bonds, padding, labels = collate_pairs([self.pair, long_pair])

        self.assertEqual(features.shape, (2, 61, 75))
        self.assertEqual((~padding).sum(1).tolist(), [3, 61])
        self.assertEqual(labels.tolist(), [2, 0])
        torch.testing.assert_close(bonds[0, :3, :3], torch.tensor(
            [[1., 1., 0.], [1., 1., 0.], [0., 0., 1.]]
        ))
        torch.testing.assert_close(bonds.diagonal(dim1=1, dim2=2), (~padding).float())
        self.assertEqual(bonds[1, :60, 60].count_nonzero().item(), 0)
        self.assertEqual(bonds[1, 60, :60].count_nonzero().item(), 0)

    def test_other_pair_padding_does_not_change_prediction(self):
        """同一对药单独评估或和更长的药对一起评估，分数应一致。"""
        self.model.eval()
        alone = collate_pairs([self.pair])
        mixed = collate_pairs([self.pair, (molecule_graph("CCCO"), molecule_graph("CN"), 0)])
        with torch.inference_mode():
            scores = self.model(*alone[:3])
            padded_scores = self.model(*mixed[:3])
            torch.testing.assert_close(scores[0], padded_scores[0], rtol=1e-5, atol=1e-5)
            _, predictions = evaluate(self.model, [alone], "cpu")
        self.assertEqual(predictions, scores.argmax(1).tolist())

    def test_qk_matches_multihead_attention_weights(self):
        """相同 Q/K 参数下，对照 PyTorch 的多头边权，覆盖补位和训练时的 dropout。"""
        reference = torch.nn.MultiheadAttention(8, 2, dropout=0.25, batch_first=True)
        model = self.model
        model.edge_dropout.p = reference.dropout
        with torch.no_grad():
            reference.in_proj_bias.normal_()
            model.edge_qk.weight.copy_(reference.in_proj_weight[:16])
            model.edge_qk.bias.copy_(reference.in_proj_bias[:16])

        atoms = torch.randn(2, 5, 8, requires_grad=True)
        graph_inputs = atoms.detach()
        padding = torch.tensor([[False, False, True, True, True], [False] * 5])
        bonds = torch.diag_embed((~padding).float())
        padded_edges = padding.unsqueeze(1) | padding.unsqueeze(2)
        upstream = torch.randn_like(bonds)
        for training in (False, True):
            with self.subTest(training=training):
                reference.train(training)
                model.train(training)
                torch.manual_seed(31)
                _, weights = reference(graph_inputs, graph_inputs, graph_inputs, key_padding_mask=padding)
                # 混合系数初始值为 0.5，补位行列都应为零。
                expected = ((bonds + weights) / 2).masked_fill(padded_edges, 0)
                torch.manual_seed(31)
                actual = model.joint_adjacency(atoms, bonds, padding)
                torch.testing.assert_close(actual, expected)

                model.zero_grad()
                reference.zero_grad()
                actual.backward(upstream)
                self.assertIsNone(atoms.grad)
                expected.backward(upstream)
                torch.testing.assert_close(model.edge_qk.weight.grad, reference.in_proj_weight.grad[:16])
                torch.testing.assert_close(model.edge_qk.bias.grad, reference.in_proj_bias.grad[:16])

    def test_single_pair_training(self):
        """最后一个药对也参与训练，梯度能回到建图和原子编码。"""
        self.model.train()
        self.assertIsInstance(self.model.classifier[2], torch.nn.LayerNorm)
        features, bonds, padding, labels = collate_pairs([self.pair])
        scores = self.model(features, bonds, padding)
        self.assertEqual(scores.shape, (1, 3))
        batch_loss(scores, labels).backward()
        for weight in (
            self.model.atom_encoder[0].weight,
            self.model.graph_mix_logit,
            self.model.classifier[2].weight,
        ):
            self.assertIsNotNone(weight.grad)
            self.assertTrue(torch.isfinite(weight.grad).all())
            self.assertGreater(weight.grad.abs().sum().item(), 0)
