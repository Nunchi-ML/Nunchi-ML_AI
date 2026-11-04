"""Copy a reproducible blind sample without rebuilding the existing frame pack."""
import argparse
import csv
import hashlib
import json
import random
import shutil
import zipfile
from pathlib import Path


def prepare(root, out, count=50, seed=47):
    root, out = root.resolve(), out.resolve()
    pack = root / "label_pack"
    if not out.is_relative_to(root) or out == root or out.is_relative_to(pack):
        raise ValueError("Review output must be a separate folder under the private data root")
    if out.exists():
        raise FileExistsError("Review folder already exists; preserve the original sample and answers")
    zip_path = out.with_suffix(".zip")
    method_path = out.with_name(out.name + "_method.json")
    if zip_path.exists() or method_path.exists():
        raise FileExistsError("Review artifacts already exist; do not overwrite them")
    sheet_path = pack / "label_sheet_blind.csv"
    before = sheet_path.read_bytes()
    with sheet_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len({r["id"] for r in rows}) != len(rows) or any(not r["id"] for r in rows):
        raise ValueError("Expected unique original frame IDs")
    if not 1 <= count <= len(rows):
        raise ValueError("Sample size is out of range")
    # Sample uses IDs/order only. Never open AI labels, predictions or source titles.
    selected = random.Random(seed).sample(sorted(rows, key=lambda r: r["id"]), count)
    paths = [(pack / r["file"]).resolve() for r in selected]
    if any(not p.is_relative_to(pack) or not p.is_file() for p in paths):
        raise ValueError("Missing/unsafe source image path")
    tool = Path(__file__).resolve().parent.parent / "glare_video_source/label_tool.html"
    out.mkdir(parents=True)
    hashes = {}
    for r, source in zip(selected, paths):
        target = out / r["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[r["id"]] = hashlib.sha256(source.read_bytes()).hexdigest()
    fields = ["no", "id", "file", "sequence", "label", "confidence", "note"]
    with (out / "label_sheet_blind.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(no=i, id=r["id"], file=r["file"], sequence=r["sequence"],
                              label="", confidence="", note="") for i, r in enumerate(selected, 1))
    blind_rows = [dict(no=i, id=r["id"], file=r["file"], sequence=r["sequence"])
                  for i, r in enumerate(selected, 1)]
    html = tool.read_text(encoding="utf-8").replace(
        '<script id="embeddedRows" type="application/json">[]</script>',
        '<script id="embeddedRows" type="application/json">' + json.dumps(blind_rows, ensure_ascii=True) + '</script>')
    (out / "label_tool.html").write_text(html, encoding="utf-8")
    (out / "README.txt").write_text(
        "사람 판정과 AI 임시 라벨의 일치도 확인용 블라인드 표본입니다. 전체 시험의 사람 2인 검증을 대신하지 않습니다.\n"
        "1. ZIP은 압축을 푼 다음 label_tool.html을 Chrome/Edge로 여세요. 사진은 바로 보입니다.\n"
        "2. 본인 이름을 적고 판정자 적용을 누르세요. 사진을 따로 업로드할 필요가 없습니다.\n"
        "3. 1=정상, 2=눈부심, 3=심한 눈부심, X=제외, U=애매함.\n"
        "4. 해/해 번짐 없이 하늘만 하얀 사진은 0. 태양광 때문에 도로 정보가 현저히 안 보일 때 2.\n"
        "5. CSV 저장 후 labels_이름.csv를 분석자에게 전달하세요. 다른 판정자의 답안은 보지 마세요.\n"
        "사진 ID와 파일을 바꾸지 마세요. 이름이 같으면 이 브라우저에서 저장한 본인 답안이 복원됩니다.\n",
        encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, arcname=f"{out.name}/{p.relative_to(out).as_posix()}")
    method = dict(seed=seed, count=count, source_rows=len(rows), selection="uniform random sample of sorted original IDs; no labels or scores",
                  sheet_sha256=hashlib.sha256(before).hexdigest(), tool_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),
                  image_sha256=hashes, zip_sha256=hashlib.sha256(zip_path.read_bytes()).hexdigest())
    method_path.write_text(json.dumps(method, ensure_ascii=False, indent=2), encoding="utf-8")
    if sheet_path.read_bytes() != before:
        raise AssertionError("Original blind sheet changed")
    print(f"{len(selected)} blind frames: {out}\nZIP: {zip_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    prepare(args.root, args.out, args.count, args.seed)
