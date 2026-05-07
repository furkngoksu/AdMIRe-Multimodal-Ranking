from __future__ import annotations

import torch

from src import data
from src.candidate_fusion import GatedCandidateFusion
from src.caption_encoder import CaptionEncoder
from src.config import get_device
from src.idiom_signal import IdiomSignal
from src.image_encoder import ImageEncoder
from src.query_encoder import QueryEncoder
from src.query_fusion import QueryFusion
from src.ranker import CosineRanker, build_prediction_records


def main() -> None:
    device = get_device()
    loader = data.get_dataloader()

    query_encoder = QueryEncoder().to(device).eval()
    caption_encoder = CaptionEncoder().to(device).eval()
    image_encoder = ImageEncoder().to(device).eval()
    idiom_signal = IdiomSignal().to(device).eval()
    candidate_fusion = GatedCandidateFusion().to(device).eval()
    query_fusion = QueryFusion().to(device).eval()
    ranker = CosineRanker().to(device).eval()

    with torch.no_grad():
        for batch in loader:
            # Query side: sentence + compound -> Q, then M adjusts Q.
            q = query_encoder(batch["sentences"], batch["compounds"])
            m = idiom_signal(batch["sentences"], batch["compounds"])
            q_prime = query_fusion(q, m)

            # Candidate side: captions and images become candidate vectors.
            t = caption_encoder(batch["captions"])
            i = image_encoder(batch["image_paths"])
            c = candidate_fusion(i, t)

            # Ranking side: compare Q' against each candidate vector.
            scores = ranker(q_prime, c)
            records = build_prediction_records(
                sample_ids=batch["sample_ids"],
                candidate_names=batch["candidate_names"],
                scores=scores,
            )

            for record in records:
                print(record)

            break


if __name__ == "__main__":
    main()
