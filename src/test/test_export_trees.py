#!/usr/bin/env python3
"""Test loading a .trees file, sampling 100 trees, and exporting a new .trees file.

Verifies that the exported file:
- Is a valid NEXUS file with taxa and trees blocks
- Contains the correct Translate mapping
- Has exactly the sampled number of tree lines
- Tree lines are valid (parseable newick strings)
- Can be re-loaded into a fresh manager and round-trips correctly
"""

import os
import sys
import re

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from treetracer.db.tree_manager import TreeManagerPandas
from treetracer.db.process_trees import process_nexus_trees_streaming


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

passed = 0
failed = 0


def check(condition, description):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {description}")
    else:
        failed += 1
        print(f"  FAIL: {description}")


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f"{'=' * 70}")


def count_tree_lines(filepath):
    """Count lines starting with 'tree ' in a file."""
    count = 0
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().startswith('tree '):
                count += 1
    return count


def parse_exported_file(filepath):
    """Parse an exported .trees file and return structural info."""
    info = {
        'has_nexus_header': False,
        'has_taxa_block': False,
        'has_trees_block': False,
        'has_translate': False,
        'has_end': False,
        'translate_entries': 0,
        'tree_count': 0,
        'tree_names': [],
        'taxa_count': 0,
        'taxa_names': [],
    }

    with open(filepath, 'r') as f:
        content = f.read()

    # Check header
    info['has_nexus_header'] = content.startswith('#NEXUS')

    # Check taxa block
    taxa_match = re.search(r'Begin taxa;.*?End;', content, re.DOTALL | re.IGNORECASE)
    if taxa_match:
        info['has_taxa_block'] = True
        # Count taxa
        taxlabels_match = re.search(r'Taxlabels\s*\n(.*?)\n\s*;',
                                     taxa_match.group(0), re.DOTALL | re.IGNORECASE)
        if taxlabels_match:
            taxa_lines = [l.strip() for l in taxlabels_match.group(1).split('\n') if l.strip()]
            info['taxa_count'] = len(taxa_lines)
            info['taxa_names'] = taxa_lines

    # Check trees block
    trees_match = re.search(r'Begin trees;', content, re.IGNORECASE)
    if trees_match:
        info['has_trees_block'] = True

    # Check Translate
    translate_match = re.search(r'Translate\s*\n(.*?)\n\s*;', content,
                                re.DOTALL | re.IGNORECASE)
    if translate_match:
        info['has_translate'] = True
        entries = [l.strip().rstrip(',') for l in translate_match.group(1).split('\n')
                   if l.strip()]
        info['translate_entries'] = len(entries)

    # Count tree lines
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped.startswith('tree '):
            eq_pos = stripped.find(' = ')
            if eq_pos > 0:
                info['tree_count'] += 1
                # Extract tree name
                left = stripped[5:eq_pos]
                name = re.sub(r'\[[^\]]*\]', '', left).strip().split()[0]
                info['tree_names'].append(name)

    # Check for End;
    info['has_end'] = bool(re.search(r'^End;\s*$', content, re.MULTILINE))

    return info


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def main():
    global passed, failed

    print("TreeTracer Export Test — Load, Sample, Export .trees file")
    print("=" * 70)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(current_dir, "test.trees")
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return 1

    file_source = os.path.basename(test_file)
    sample_size = 100

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    print_section("1. LOAD")

    mgr = TreeManagerPandas()
    count = process_nexus_trees_streaming(test_file, mgr, file_source,
                                          batch_size=500, transaction_size=1000)
    mgr.flush()

    check(count > 0, f"Loaded {count} trees from {file_source}")

    # Check preamble was captured
    preamble = mgr.get_source_preamble(file_source)
    check(preamble is not None, "Preamble was captured")
    if preamble:
        preamble_text = preamble.decode('utf-8', errors='replace')
        print(f"  Preamble size: {len(preamble)} bytes")
        check('#NEXUS' in preamble_text, "Preamble contains #NEXUS header")
        check('Begin taxa;' in preamble_text or 'begin taxa;' in preamble_text.lower(),
              "Preamble contains taxa block")
        check('Translate' in preamble_text, "Preamble contains Translate section")

    # Check translate map
    translate_map = mgr.get_translate_map(file_source)
    check(translate_map is not None and len(translate_map) > 0,
          f"Translate map parsed with {len(translate_map or {})} entries")
    if translate_map:
        # Check a few entries
        check('1' in translate_map, "Translate map contains key '1'")
        first_val = translate_map.get('1', '')
        print(f"  Translate['1'] = {first_val}")
        last_key = str(len(translate_map))
        check(last_key in translate_map, f"Translate map contains key '{last_key}'")
        print(f"  Translate['{last_key}'] = {translate_map.get(last_key, '')}")

    # ------------------------------------------------------------------
    # 2. Sample
    # ------------------------------------------------------------------
    print_section("2. SAMPLE")

    trees = mgr.get_trees_sample(
        filters={'file_source': file_source},
        limit=sample_size,
        strategy='random'
    )

    check(len(trees) == sample_size, f"Sampled {len(trees)} trees (expected {sample_size})")

    # Verify sampled trees have newick content
    all_have_newick = all(len(t['newick']) > 0 for t in trees)
    check(all_have_newick, "All sampled trees have non-empty newick strings")

    # Verify tree names are present
    tree_names = [t['name'] for t in trees]
    check(all(n for n in tree_names), "All sampled trees have non-empty names")
    print(f"  Sample tree names (first 5): {tree_names[:5]}")

    # ------------------------------------------------------------------
    # 3. Export
    # ------------------------------------------------------------------
    print_section("3. EXPORT")

    output_path = os.path.join(current_dir, "test_export.trees")

    try:
        written = mgr.export_trees_nexus(output_path, trees)
        check(written == sample_size, f"Exported {written} trees (expected {sample_size})")

        output_size = os.path.getsize(output_path)
        print(f"  Output file: {output_path}")
        print(f"  Output size: {output_size / 1024:.1f} KB")

        # ------------------------------------------------------------------
        # 4. Validate exported file structure
        # ------------------------------------------------------------------
        print_section("4. VALIDATE EXPORTED FILE STRUCTURE")

        info = parse_exported_file(output_path)

        check(info['has_nexus_header'], "Exported file has #NEXUS header")
        check(info['has_taxa_block'], "Exported file has taxa block")
        check(info['has_trees_block'], "Exported file has trees block")
        check(info['has_translate'], "Exported file has Translate section")
        check(info['has_end'], "Exported file ends with 'End;'")
        check(info['tree_count'] == sample_size,
              f"Exported file has {info['tree_count']} tree lines (expected {sample_size})")

        if translate_map:
            check(info['translate_entries'] == len(translate_map),
                  f"Translate entries: {info['translate_entries']} "
                  f"(expected {len(translate_map)})")
            check(info['taxa_count'] == len(translate_map),
                  f"Taxa count: {info['taxa_count']} (expected {len(translate_map)})")

        # Verify exported tree names match sampled tree names.
        # The manager prefixes names with "group/" so strip that for comparison.
        exported_states = {re.search(r'STATE_\d+', n).group() for n in info['tree_names']
                          if re.search(r'STATE_\d+', n)}
        sampled_states = {re.search(r'STATE_\d+', n).group() for n in tree_names
                         if re.search(r'STATE_\d+', n)}
        check(exported_states == sampled_states,
              f"Exported tree STATE ids match sampled tree STATE ids "
              f"({len(exported_states)} exported, {len(sampled_states)} sampled)")

        # Verify trees are in MCMC order (STATE numbers should be ascending)
        state_numbers = []
        for name in info['tree_names']:
            m = re.search(r'STATE_(\d+)', name)
            if m:
                state_numbers.append(int(m.group(1)))
        check(len(state_numbers) == sample_size,
              f"All {len(state_numbers)} tree names have STATE numbers")
        check(state_numbers == sorted(state_numbers),
              "Exported trees are in MCMC order (ascending STATE numbers)")
        if state_numbers:
            print(f"  First: STATE_{state_numbers[0]}, Last: STATE_{state_numbers[-1]}")

        # ------------------------------------------------------------------
        # 5. Re-load exported file and verify round-trip
        # ------------------------------------------------------------------
        print_section("5. RE-LOAD EXPORTED FILE (round-trip)")

        mgr2 = TreeManagerPandas()
        export_source = os.path.basename(output_path)
        count2 = process_nexus_trees_streaming(output_path, mgr2, export_source,
                                                batch_size=500, transaction_size=1000)
        mgr2.flush()

        check(count2 == sample_size,
              f"Re-loaded {count2} trees from exported file (expected {sample_size})")

        # Verify preamble round-trips
        preamble2 = mgr2.get_source_preamble(export_source)
        check(preamble2 is not None, "Re-loaded file has preamble")
        if preamble and preamble2:
            check(preamble == preamble2,
                  "Preamble round-trips exactly (byte-identical)")

        # Verify translate map round-trips
        translate2 = mgr2.get_translate_map(export_source)
        check(translate2 == translate_map,
              f"Translate map round-trips exactly ({len(translate2 or {})} entries)")

        # Verify newick strings round-trip
        trees2 = mgr2.get_trees_sample(limit=sample_size, strategy='uniform')
        check(len(trees2) == sample_size,
              f"Re-sampled {len(trees2)} trees from re-loaded file")

        # Compare newick strings by STATE id (names have different group prefixes
        # since the original file_source and export file_source differ).
        def state_id(name):
            m = re.search(r'STATE_\d+', name)
            return m.group() if m else name

        original_by_state = {state_id(t['name']): t['newick'] for t in trees}
        reloaded_by_state = {state_id(t['name']): t['newick'] for t in trees2}

        matching_newicks = 0
        for sid in original_by_state:
            if sid in reloaded_by_state:
                if original_by_state[sid] == reloaded_by_state[sid]:
                    matching_newicks += 1

        check(matching_newicks == sample_size,
              f"All {matching_newicks}/{sample_size} newick strings match after round-trip")

        mgr2.cleanup()

    finally:
        pass  # Keep test_export.trees for inspection

    mgr.cleanup()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_section("SUMMARY")
    total = passed + failed
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")

    if failed > 0:
        print(f"\n  {failed} TEST(S) FAILED")
        return 1
    else:
        print(f"\n  ALL {passed} TESTS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
