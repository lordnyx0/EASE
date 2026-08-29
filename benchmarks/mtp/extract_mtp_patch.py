import re

PATCH_PATH = r"patches\llama-cpp-dcfr-research.patch"

with open(PATCH_PATH, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

print("=" * 85)
print(" [EXTRAÇÃO DO COMMON_SPECULATIVE_IMPL_DRAFT_MTP NO PATCH KADENBALL]")
print("=" * 85)

recording = False
extracted = []

for idx, line in enumerate(lines):
    if "common_speculative_impl_draft_mtp" in line or "spec_mtp" in line:
        start = max(0, idx - 10)
        end = min(len(lines), idx + 80)
        for j in range(start, end):
            extracted.append(lines[j])
        break

print("".join(extracted[:120]))
print("=" * 85)
