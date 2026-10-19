"""Zero-based landmark conventions. Left/right always refer to the displayed image."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Topology:
    name: str
    count: int
    outline: tuple[int, ...]
    eyes: tuple[tuple[int, ...], tuple[int, ...]]
    brows: tuple[tuple[int, ...], tuple[int, ...]]
    corners: tuple[tuple[int, int], tuple[int, int]]  # inner, outer
    lids: tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]
    width: tuple[int, int]
    jaw: tuple[int, int]
    chin: int
    bridge: int
    nose: int
    normalizer: tuple[int, int]
    pairs: tuple[tuple[int, int], ...]
    paths: tuple[tuple[int, ...], ...]
    flip: tuple[int, ...]


def make_topology(name: str) -> Topology:
    if name == "300w":
        pairs = ([(i, 16-i) for i in range(8)] + [(i, 43-i) for i in range(17, 22)]
                 + [(31,35),(32,34),(36,45),(37,44),(38,43),(39,42),(40,47),(41,46),
                    (48,54),(49,53),(50,52),(55,59),(56,58),(60,64),(61,63),(65,67)])
        flip = list(range(68))
        for a,b in pairs:
            flip[a],flip[b] = b,a
        return Topology(name,68,tuple(range(17)),(tuple(range(36,42)),tuple(range(42,48))),
                        (tuple(range(17,22)),tuple(range(22,27))),((39,36),(42,45)),
                        (((37,41),(38,40)),((43,47),(44,46))), (0,16),(4,12),8,27,33,(36,45),
                        tuple(pairs), (tuple(range(17)),tuple(range(17,22)),tuple(range(22,27)),
                        tuple(range(27,31)),tuple(range(31,36)),(36,37,38,39,40,41,36),
                        (42,43,44,45,46,47,42),tuple(range(48,60))+(48,),tuple(range(60,68))+(60,)),tuple(flip))
    if name == "wflw":
        pairs = ([(i,32-i) for i in range(16)] + list(zip(range(33,42),[46,45,44,43,42,50,49,48,47]))
                 + [(55,59),(56,58),(60,72),(61,71),(62,70),(63,69),(64,68),(65,75),(66,74),(67,73),
                    (76,82),(77,81),(78,80),(83,87),(84,86),(88,92),(89,91),(93,95),(96,97)])
        flip = list(range(98))
        for a,b in pairs:
            flip[a],flip[b] = b,a
        return Topology(name,98,tuple(range(33)),(tuple(range(60,68)),tuple(range(68,76))),
                        (tuple(range(33,38)),tuple(range(42,47))),((64,60),(68,72)),
                        (((61,67),(62,66),(63,65)),((69,75),(70,74),(71,73))), (0,32),(8,24),16,51,57,(96,97),
                        tuple(pairs),(tuple(range(33)),tuple(range(33,42))+(33,),tuple(range(42,51))+(42,),
                        tuple(range(51,55)),tuple(range(55,60)),tuple(range(60,68))+(60,),
                        tuple(range(68,76))+(68,),tuple(range(76,88))+(76,),tuple(range(88,96))+(88,)),tuple(flip))
    raise ValueError("Dataset must be 'wflw' or '300w'.")
