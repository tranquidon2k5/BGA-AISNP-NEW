#!/usr/bin/env python3
import csv
from pathlib import Path

# ==== CONFIG ====
BASE = Path.home() / "Documents" / "2025.1" / "Bio_Labs" / "58AISNPs"
OUTDIR = BASE / "1kgp_58AISNPs_out"
OUTDIR.mkdir(parents=True, exist_ok=True)

FILE_MATRIX = OUTDIR / "AISNP_GT_matrix_with_refalt.tsv"
FILE_CSVLIST = BASE / "AISNP_list.csv"
FILE_PANEL = BASE / "integrated_call_samples_v3.20130502.ALL.panel.txt"

OUT_CONT = OUTDIR / "AISNP_by_sample_continental.csv"
OUT_EAS = OUTDIR / "AISNP_by_sample_eastasian.csv"


def detect_delim(path):
    head = path.read_text(encoding="utf-8").splitlines()[0]
    if ";" in head:
        return ";"
    elif "," in head:
        return ","
    return ","


def load_panel(panel_path):
    """
    panel gốc của 1KGP thường có cột:
    sample  population  super_population  gender
    mình map sample -> (pop, super_pop)
    """
    sample_info = {}
    with panel_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sample = row.get("sample") or row.get("Sample") or row.get("sample_id")
            pop = row.get("population") or row.get("pop") or ""
            sp = row.get("super_population") or row.get("super_pop") or ""
            if sample:
                sample_info[sample] = (pop, sp)
    return sample_info


def load_aisnp_list(csv_path):
    """
    Trả về:
      - order_all: list rsID theo đúng thứ tự file
      - cont_ids: list rsID continental
      - ea_ids: list rsID East Asian
      - locus_map: rsID -> type
    """
    delim = detect_delim(csv_path)
    order_all = []
    cont_ids = []
    ea_ids = []
    locus_map = {}

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            rs = row.get("AISNP") or row.get("rsid") or row.get("rsID")
            locus = row.get("Ancestry-specific locus") or row.get("locus") or ""
            if not rs:
                continue
            rs = rs.strip()
            locus_lower = locus.strip().lower()
            order_all.append(rs)
            locus_map[rs] = locus
            if "east asian" in locus_lower:
                ea_ids.append(rs)
            elif "continental" in locus_lower:
                cont_ids.append(rs)
            else:
                # nếu không rõ thì cứ cho vào continental để không mất
                cont_ids.append(rs)
    return order_all, cont_ids, ea_ids, locus_map


def load_matrix(matrix_path):
    """
    Đọc file TSV dạng:
    CHROM POS ID REF ALT sample1 sample2 ...
    Trả về:
      samples: list tên sample theo đúng thứ tự cột
      snp_data: dict rsID -> {
          "ref": ref,
          "alt": alt,
          "gts": [gt_sample1, gt_sample2, ...]
      }
    """
    snp_data = {}
    with matrix_path.open() as f:
      # đọc dòng đầu
        header = f.readline().rstrip("\n").split("\t")
        # 0:CHROM 1:POS 2:ID 3:REF 4:ALT -> từ 5 trở đi là sample
        samples = header[5:]

        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            chrom, pos, rsid, ref, alt = parts[:5]
            gts = parts[5:]
            snp_data[rsid] = {
                "ref": ref,
                "alt": alt,
                "gts": gts,
            }

    return samples, snp_data


def gt_to_alleles(gt, ref, alt):
    """
    0|1 -> (ref, alt)
    1|0 -> (alt, ref)
    0|0 -> (ref, ref)
    1|1 -> (alt, alt)
    . hoặc ./., .|. -> ("NA", "NA")
    """
    if gt is None or gt == "" or gt.startswith("."):
        return ("NA", "NA")
    if "|" in gt:
        a, b = gt.split("|", 1)
    elif "/" in gt:
        a, b = gt.split("/", 1)
    else:
        # trường hợp hiếm
        return ("NA", "NA")

    def map_allele(x):
        if x == "0":
            return ref
        elif x == "1":
            return alt
        else:
            # phòng trường hợp có allele 2,3... trong multi-allelic nhưng vcf này biallelic
            return "NA"

    return (map_allele(a), map_allele(b))


def write_csv(out_path, samples, snp_data, rs_order, sample_panel):
    """
    out: sample,pop,super_pop, rs1_1,rs1_2, rs2_1,rs2_2...
    """
    # build header
    header = ["sample", "pop", "super_pop"]
    for rs in rs_order:
        header.append(f"{rs}_1")
        header.append(f"{rs}_2")

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for idx, samp in enumerate(samples):
            pop, sp = sample_panel.get(samp, ("", ""))
            row = [samp, pop, sp]
            for rs in rs_order:
                if rs not in snp_data:
                    # SNP này không extract được từ 1KGP
                    row.extend(["NA", "NA"])
                else:
                    info = snp_data[rs]
                    gt_list = info["gts"]
                    # idx của sample tương ứng vị trí của nó trong header
                    if idx < len(gt_list):
                        gt = gt_list[idx]
                    else:
                        gt = ""
                    a1, a2 = gt_to_alleles(gt, info["ref"], info["alt"])
                    row.extend([a1, a2])
            writer.writerow(row)


def main():
    # 1) load panel
    sample_panel = load_panel(FILE_PANEL)
    # 2) load aisnp list
    order_all, cont_ids, ea_ids, locus_map = load_aisnp_list(FILE_CSVLIST)
    # 3) load matrix
    samples, snp_data = load_matrix(FILE_MATRIX)

    # 4) ghi 2 file
    print(f"[+] Writing continental file to {OUT_CONT}")
    write_csv(OUT_CONT, samples, snp_data, cont_ids, sample_panel)

    print(f"[+] Writing East Asian file to {OUT_EAS}")
    write_csv(OUT_EAS, samples, snp_data, ea_ids, sample_panel)

    print("[✅] Done.")
    print(f"  Continental SNPs: {len(cont_ids)}")
    print(f"  East Asian SNPs: {len(ea_ids)}")
    print(f"  Samples: {len(samples)}")


if __name__ == "__main__":
    main()