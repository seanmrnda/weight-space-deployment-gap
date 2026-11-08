"""Measure which experts each domain routes to, on real forwards of one recipe.

Per sparse block and source, reports the share of top-k expert selections and
mean routing entropy. Experts that specialize by domain would allow an edit
narrower than a whole expert group. Reads the plan and writes one JSON receipt.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2048_weight_space as wsp
import p2048_weight_space_soup as soup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-source', type=int, default=40)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = soup.panel_rows('screen')
    model_args = argparse.Namespace(model=wsp.DEFAULT_MODEL, expected_model_sha256=wsp.DEPLOYED_SHA256,
                                    threads=args.threads, device='cpu')
    torch, model, collate, _ = wsp.runtime(model_args)
    blocks = model.moe_blocks
    layers = model.spec['moe_layers']
    report = dict(schema='p2048-weight-space-routing-v1', recipe='L', per_source=args.per_source, moe_layers=layers,
                  top_k=model.spec['top_k'], num_experts=model.spec['num_experts'], sources={})
    for block in blocks:
        block.observe_routing = True
    for source in soup.SOURCES:
        for block in blocks:
            block.routing_stats = None
        chosen = [r for r in rows if r['source_family'] == source][:args.per_source]
        for row in chosen:
            request, _ = wsp.rotate(row, 0)
            with torch.inference_mode():
                model(**collate([request]))
        per_layer = {}
        for layer, block in zip(layers, blocks):
            stats = block.routing_stats
            total = sum(stats['selection_counts'])
            per_layer[str(layer)] = dict(share=[round(c / total, 4) for c in stats['selection_counts']],
                                         mean_entropy=round(stats['entropy_sum'] / stats['valid_tokens'], 4),
                                         tokens=stats['valid_tokens'])
        report['sources'][source] = per_layer
    wsp.write(args.output, report)
    for layer in map(str, layers):
        print(layer, ' | '.join(f"{s[:2]} {report['sources'][s][layer]['share']}" for s in soup.SOURCES))


if __name__ == '__main__':
    main()
