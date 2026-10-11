#!/usr/bin/env python3
"""Static release preflight; does not validate MATLAB numerics."""
from pathlib import Path
import re,sys,argparse
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--final', action='store_true', help='require paper-linked v1.0.0 artifacts')
args=parser.parse_args()
required=[
    'README.md','LICENSE','CITATION.cff','VERSION','config/default_config.m',
    'config/final_cssp_v1.m','config/method_manifest.csv',
    'docs/PUBLICATION_SUPPLEMENT_PLAN_CN.md','docs/EXPERIMENT_PROTOCOL_V1_CN.md',
    'docs/RELEASE_CHECKLIST_CN.md','docs/raw_result_schema.md',
    'docs/paper_table_figure_map.csv','docs/method_map.csv',
    'data/dataset_manifest.csv','paper/README.md','supplementary/README.md',
    'results/final/README.md','experiments/run_selftest.m'
]
missing=[x for x in required if not (root/x).exists()]
active=list((root/'src').rglob('*.m'))+list((root/'config').rglob('*.m'))+list((root/'experiments').glob('*.m'))
abs_hits=[]
for p in active:
    text=p.read_text(errors='ignore')
    if re.search(r'(?i)([A-Z]:\\|/Users/|/home/)',text): abs_hits.append(str(p.relative_to(root)))
placeholder_hits=[]
for rel in ['paper/README.md','supplementary/README.md','docs/paper_table_figure_map.csv','data/dataset_manifest.csv']:
    text=(root/rel).read_text(errors='ignore')
    if 'pending' in text.lower() or 'placeholder' in text.lower(): placeholder_hits.append(rel)
print('required_files:', 'PASS' if not missing else 'FAIL')
if missing: print('missing:', *missing, sep='\n  ')
print('active_absolute_paths:', 'PASS' if not abs_hits else 'FAIL')
if abs_hits: print('absolute-path-hits:', *abs_hits, sep='\n  ')
print('publication_templates_pending:', ', '.join(placeholder_hits) if placeholder_hits else 'none')
legacy_hits=[]
for p in (root/'experiments/legacy').glob('*.m'):
    if re.search(r'(?i)([A-Z]:\\|/Users/|/home/)', p.read_text(errors='ignore')): legacy_hits.append(str(p.relative_to(root)))
print('legacy_absolute_paths:', 'documented' if legacy_hits else 'none')
final_missing=[]
if args.final:
    for rel in ['paper/L1RBPCA_CSSP_Final.tex','paper/references.bib','supplementary/L1RBPCA_CSSP_Supplementary.tex']:
        if not (root/rel).exists(): final_missing.append(rel)
    if not any((root/'results/final').rglob('raw*.csv')):
        final_missing.append('results/final/raw*.csv')
    print('final_v1_artifacts:', 'PASS' if not final_missing else 'FAIL')
    if final_missing: print('final_missing:', *final_missing, sep='\n  ')
print('NOTE: MATLAB execution, data licenses, algorithm invariants, and paper-result reproduction require the final v1.0.0 rerun.')
sys.exit(1 if missing or abs_hits or final_missing else 0)
