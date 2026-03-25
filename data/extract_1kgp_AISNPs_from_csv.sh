#!/usr/bin/env bash
# ============================================================
#  Script: extract_1kgp_AISNPs_from_csv.sh
#  Purpose: Đọc danh sách AISNP từ file CSV (AISNP_list.csv),
#           tách theo chromosome và trích toàn bộ genotype của
#           các mẫu 1KGP (Phase 3, hg19) kèm REF/ALT.
#  Output : .../AISNP_GT_matrix_with_refalt.tsv
#  Note   : Cần bcftools, curl. Hợp với macOS bash 3.2.
# ============================================================

set -euo pipefail
trap 'echo "❌ Error at line $LINENO (exit code $?)."' ERR

# ==== CONFIG (sửa lại đường dẫn gốc của bạn ở đây) ====
BASE="${HOME}/Documents/2025.1/Bio_Labs/58AISNPs"   # thư mục chứa AISNP_list.csv
CSVFILE="${BASE}/AISNP_list.csv"

OUTDIR="${BASE}/1kgp_58AISNPs_out"
TMPDIR="${BASE}/1kgp_58AISNPs_tmp"
mkdir -p "${OUTDIR}" "${TMPDIR}"

MIRROR1="https://ftp-trace.ncbi.nlm.nih.gov/1000genomes/ftp/release/20130502"
MIRROR2="https://ftp.ebi.ac.uk/pub/databases/1000genomes/ftp/release/20130502"

OUTFILE="${OUTDIR}/AISNP_GT_matrix_with_refalt.tsv"
HDRFILE="${OUTDIR}/samples_header.tsv"

# ==== CHECK INPUT ====
if [ ! -s "${CSVFILE}" ]; then
  echo "❌ Không tìm thấy file CSV: ${CSVFILE}"
  exit 1
fi

echo "[+] Using CSV: ${CSVFILE}"

# ==== STEP 0: Đoán delimiter (',' hay ';') ====
DELIM=$(head -n1 "${CSVFILE}" | awk 'index($0,";")>0{print ";"} index($0,",")>0{print ","}')
if [ -z "${DELIM}" ]; then
  echo "❌ Không đoán được delimiter của CSV (không thấy , hoặc ;)"
  exit 1
fi
echo "[+] Detected delimiter: '${DELIM}'"

# ==== STEP 1: Tách rsID theo chromosome ====
# Ta sẽ tạo các file: ${TMPDIR}/rs_chr1.txt, rs_chr2.txt, ... theo đúng chr trong CSV
# Giả định CSV có cột tên 'AISNP' và 'chromosome'
echo "[+] Parsing CSV để tách AISNP theo chromosome..."
# Xoá các file cũ
rm -f "${TMPDIR}"/rs_chr*.txt

# Lấy index cột AISNP và chromosome (phòng trường hợp thứ tự cột đổi)
header=$(head -n1 "${CSVFILE}")
# chuyển header thành mảng
IFS="${DELIM}" read -r -a cols <<< "${header}"

idx_rs=-1
idx_chr=-1
for i in "${!cols[@]}"; do
  colname=$(echo "${cols[$i]}" | tr -d '\r"')
  if [ "${colname}" = "AISNP" ]; then
    idx_rs=$((i+1))
  elif [[ "${colname}" = "chromosome" || "${colname}" = "chr" ]]; then
    idx_chr=$((i+1))
  fi
done

if [ ${idx_rs} -eq -1 ] || [ ${idx_chr} -eq -1 ]; then
  echo "❌ Không tìm thấy cột 'AISNP' hoặc 'chromosome' trong CSV"
  echo "   Header: ${header}"
  exit 1
fi

# Đọc từng dòng và ghi ra file theo chr
tail -n +2 "${CSVFILE}" | awk -v FS="${DELIM}" -v rscol=${idx_rs} -v chrcol=${idx_chr} -v outdir="${TMPDIR}" '
  NF>0 {
    gsub(/\r/,"")
    rs=$rscol
    chr=$chrcol
    # chuẩn hoá chr: bỏ "chr" nếu có
    gsub(/^chr/,"",chr)
    if (rs != "" && chr != "") {
      fname=sprintf("%s/rs_chr%s.txt", outdir, chr)
      print rs >> fname
    }
  }
'

echo "[+] Đã tạo các file rs_chr*.txt trong ${TMPDIR}"
ls -1 "${TMPDIR}"/rs_chr*.txt 2>/dev/null || true

# ==== STEP 2: Lấy danh sách sample từ chr1 ====
echo "[+] Getting 1KGP sample list from chr1..."
VCF1="${TMPDIR}/ALL.chr1.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
VCFURL1="${MIRROR1}/ALL.chr1.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"

if [ ! -s "${VCF1}" ]; then
  # không tải về file luôn để đỡ nặng, query trực tiếp qua URL
  if ! curl -fsI "${VCFURL1}" >/dev/null 2>&1; then
    VCFURL1="${MIRROR2}/ALL.chr1.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
  fi
  bcftools query -l "${VCFURL1}" > "${HDRFILE}" 2>/dev/null || {
    echo "❌ Cannot fetch sample list"
    exit 1
  }
else
  bcftools query -l "${VCF1}" > "${HDRFILE}"
fi

# Build header line cho file output
header="CHROM\tPOS\tID\tREF\tALT"
while read -r s; do
  header="${header}\t${s}"
done < "${HDRFILE}"
echo -e "${header}" > "${OUTFILE}"
echo "[+] Output initialized: ${OUTFILE}"

# ==== STEP 3: Hàm lấy suffix theo chr (v5a cho 1..22, v1b cho X) ====
get_suffix() {
  local chr="$1"
  if [ "${chr}" = "X" ]; then
    echo "v1b"
  else
    echo "v5a"
  fi
}

# ==== STEP 4: Loop qua tất cả file rs_chr*.txt vừa tạo ====
echo "[+] Bắt đầu trích dữ liệu theo chromosome..."

for rsfile in "${TMPDIR}"/rs_chr*.txt; do
  [ -e "$rsfile" ] || continue
  chr=$(basename "$rsfile" | sed 's/rs_chr//' | sed 's/.txt//')

  echo "==> chr${chr}"

  SUFFIX=$(get_suffix "${chr}")
  FN="ALL.chr${chr}.phase3_shapeit2_mvncall_integrated_${SUFFIX}.20130502.genotypes.vcf.gz"
  SRC1="${MIRROR1}/${FN}"
  SRC2="${MIRROR2}/${FN}"
  TMP="${OUTDIR}/_chr${chr}_AISNP.tmp.tsv"

  # chọn nguồn còn sống
  SRC=""
  if curl -fsI "${SRC1}" >/dev/null 2>&1; then
    SRC="${SRC1}"
  elif curl -fsI "${SRC2}" >/devnull 2>&1; then
    SRC="${SRC2}"
  else
    echo "  ⚠️ Không truy cập được chr${chr}, bỏ qua."
    continue
  fi

  # trích
  echo "  ▶ Extracting AISNP from chr${chr}..."
  if bcftools view -i "ID=@${rsfile}" "${SRC}" -Ou 2>/dev/null | \
     bcftools query -f '%CHROM\t%POS\t%ID\t%REF\t%ALT[\t%GT]\n' > "${TMP}" 2>/dev/null; then
    lines=$(wc -l < "${TMP}" | awk '{print $1}')
    echo "  ✅ Extracted ${lines} records from chr${chr}"
    cat "${TMP}" >> "${OUTFILE}"
  else
    echo "  ❌ Extraction failed for chr${chr}"
  fi
  rm -f "${TMP}"
done

# ==== STEP 5: Summary ====
echo
echo "✅ Completed extraction!"
echo "[+] Output: ${OUTFILE}"
echo "[+] SNP count (rows, trừ header): $(($(wc -l < "${OUTFILE}") - 1))"
echo "[+] Preview:"
head -n 10 "${OUTFILE}"