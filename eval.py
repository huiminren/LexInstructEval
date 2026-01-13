import json
import re
from typing import List, Dict, Any, Union, Optional, Tuple
from collections import Counter
import nltk
import time
import threading
from functools import wraps
from typing import Any, List, Dict, Tuple, Optional, Callable
import concurrent.futures
from typing import Any, List, Dict, Tuple
import multiprocessing
import signal
from contextlib import contextmanager
import os
import glob
from datetime import datetime


BIG_LEVELS = {'answer', 'paragraph', 'line', 'bullet', 'sentence'}
SMALL_LEVELS = {'letter', 'punc', 'pattern', 'character', 'word', 'phase'}

PUNCTUATION_SET = set(r"""!"$%&'(),./:;<=>?@[\]^_`{|}~，。！？：；""''（）《》【】『』「」﹃﹄〔〕—…""")

try:
    from nltk.tokenize import RegexpTokenizer
    chinese_sentence_tokenizer = RegexpTokenizer(".*?[。！？]")
except:
    chinese_sentence_tokenizer = None

class TimeoutError(Exception):
    pass

@contextmanager
def timeout_context(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def validate_single_answer(answer: Any, 
                         instruction_pattern: List[Dict[str, Any]], 
                         language: str,
                         rule_timeout: int = 3) -> Tuple[bool, Dict[str, Any]]:

    num_rules = len(instruction_pattern) if instruction_pattern else 0
    if answer is None:
        return False, {"follow_all_list": [False] * num_rules, "wrong_reason": "Answer is None"}
    if not isinstance(answer, str):
        return False, {"follow_all_list": [False] * num_rules, "wrong_reason": f"Answer must be string type, got {type(answer).__name__}"}
    if not answer.strip():
        return False, {"follow_all_list": [False] * num_rules, "wrong_reason": "Answer is empty or contains only whitespace"}
    if not instruction_pattern or not isinstance(instruction_pattern, list):
        return False, {"follow_all_list": [], "wrong_reason": "Invalid instruction_pattern: must be a non-empty list"}
    repetition_check = check_excessive_repetition(answer)
    if repetition_check["has_repetition"]:
        return False, {"follow_all_list": [False] * num_rules, "wrong_reason": repetition_check["reason"]}
    

    processed_answer = answer.strip()
    cache = build_segmentation_cache(processed_answer, language)
    

    follow_all_list = []
    all_wrong_reasons = []

    for rule_idx, rule in enumerate(instruction_pattern):
        try:
            with timeout_context(rule_timeout):
                rule_passed, rule_reason = check_rule(rule, cache, language)
                follow_all_list.append(rule_passed)
                
                if not rule_passed:
                    wrong_reason_for_this_rule = build_wrong_reason(
                        rule_idx + 1,
                        len(instruction_pattern),
                        rule,
                        rule_reason,
                        cache
                    )
                    all_wrong_reasons.append(wrong_reason_for_this_rule)
                    
        except TimeoutError:
            follow_all_list.append(False)
            timeout_reason = f"Rule {rule_idx + 1} validation timed out after {rule_timeout} seconds"
            wrong_reason_for_this_rule = build_wrong_reason(
                rule_idx + 1,
                len(instruction_pattern),
                rule,
                timeout_reason,
                cache
            )
            all_wrong_reasons.append(wrong_reason_for_this_rule)
            
        except Exception as e:
            follow_all_list.append(False)
            error_reason = f"Rule {rule_idx + 1} validation failed with error: {str(e)}"
            wrong_reason_for_this_rule = build_wrong_reason(
                rule_idx + 1,
                len(instruction_pattern),
                rule,
                error_reason,
                cache
            )
            all_wrong_reasons.append(wrong_reason_for_this_rule)

    overall_passed = all(follow_all_list)
    final_wrong_reason = "\n\n".join(all_wrong_reasons) if all_wrong_reasons else None
    
    return overall_passed, {
        "follow_all_list": follow_all_list,
        "wrong_reason": final_wrong_reason
    }


def check_excessive_repetition(text: str, ngram_size: int = 20, threshold: int = 20) -> Dict[str, Any]:

    

    char_check = check_char_repetition(text, min_repeat=150)  
    if char_check["has_repetition"]:
        return char_check
    

    words = text.split()
    if len(words) < ngram_size:
        return {"has_repetition": False, "reason": ""}
    

    ngrams = []
    for i in range(len(words) - ngram_size + 1):
        ngram = tuple(words[i:i + ngram_size])
        ngrams.append(ngram)
    

    ngram_counts = Counter(ngrams)
    

    for ngram, count in ngram_counts.items():
        if count >= threshold:
            repeated_text = " ".join(ngram)
            preview = repeated_text[:100] + "..." if len(repeated_text) > 100 else repeated_text
            
            return {
                "has_repetition": True,
                "reason": f"Excessive repetition detected: {ngram_size}-gram repeated {count} times. "
                         f"Repeated content: '{preview}'"
            }
    
    return {"has_repetition": False, "reason": ""}

def check_char_repetition(text: str, min_repeat: int = 100) -> Dict[str, Any]:

    if len(text) < min_repeat:
        return {"has_repetition": False, "reason": ""}
    
    i = 0
    while i < len(text):
        char = text[i]
        count = 1
        

        while i + count < len(text) and text[i + count] == char:
            count += 1
        
        if count >= min_repeat:
            preview = char * min(50, count)
            return {
                "has_repetition": True,
                "reason": f"Excessive character repetition: '{char}' repeated {count} times. Preview: '{preview}'"
            }
        
        i += count
    
    return {"has_repetition": False, "reason": ""}

def build_wrong_reason(rule_idx: int, total_rules: int, rule: Dict[str, Any], 
                      failure_reason: str, cache: Dict[str, List[str]]) -> str:


    procedure = rule.get('procedure', [])
    rule_path = []
    for step in procedure:
        level = step.get('level', 'unknown')
        predicate = step.get('predicate', '')
        description = step.get('description', '')
        if description:
            rule_path.append(f"{level}{predicate}('{description}')")
        else:
            rule_path.append(f"{level}{predicate}")
    

    relevant_segments = []
    if procedure:
        last_level = procedure[-1].get('level', 'answer')
        segments = cache.get(last_level, [])

        for i, seg in enumerate(segments):
            preview = seg
            relevant_segments.append(f"  [{i+1}] {preview}")
    

    wrong_reason = f"Rule {rule_idx}/{total_rules} failed:\n"
    wrong_reason += f"  Rule path: {' -> '.join(rule_path)}\n"
    wrong_reason += f"  Failure: {failure_reason}\n"
    
    if relevant_segments:
        wrong_reason += f"  Candidate segments at level '{last_level}':\n"
        wrong_reason += "\n".join(relevant_segments)
        if len(segments) > 3:
            wrong_reason += f"\n  ... and {len(segments) - 3} more segments"
    

    if 'relation' in rule and 'value' in rule:
        wrong_reason += f"\n  Expected: {rule['relation']} '{rule['value']}'"
    
    return wrong_reason




def split_text(text: str, level: str, language: str) -> List[str]:

    if not text: 
        return []
    
    if level == 'answer': 
        return [text]
        
    if level == 'paragraph': 

        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text.strip()) if p.strip()]
        return paragraphs
        
    if level == 'line': 

        lines = text.splitlines(keepends=False)
        lines = [line.rstrip() for line in lines if line.strip()] 
        return lines
        
    if level == 'bullet':
        lines = text.splitlines(keepends=False)
        bullet_pattern = r'^\s*([*+\-]|\d+\.|\w+\.)\s+(.*)'
        bullet_items = []
        for line in lines:
            match = re.match(bullet_pattern, line)
            if match:

                content = match.group(2).rstrip() if match.group(2) else ""
                if content: 
                    bullet_items.append(content)
        return bullet_items
        
    if level == 'sentence':
        if language == 'chinese' and chinese_sentence_tokenizer:
            sentences = chinese_sentence_tokenizer.tokenize(text)
        else:
            try:
                sentences = nltk.sent_tokenize(text)
            except:

                sentences = re.split(r'(?<=[.!?])\s+', text)

        sentences = [sent.strip() for sent in sentences if sent.strip()]
        return sentences
        
    if level == 'word':

        return text.split() if language == 'english' else []
        
    if level == 'character':

        return [char for char in text if '\u4e00' <= char <= '\u9fff'] if language == 'chinese' else []
        
    if level == 'letter':

        return [char for char in text if char.isalpha()] if language == 'english' else []
         
    if level == 'punc': 

        return [char for char in text if char in PUNCTUATION_SET]
        
    return []

def build_segmentation_cache(text: str, language: str) -> Dict[str, List[str]]:

    cache = {}
    stripped_text = text.strip()
    cache['answer'] = [stripped_text] if stripped_text else []
    for lvl in BIG_LEVELS:
        cache[lvl] = split_text(stripped_text, lvl, language)
    return cache


def parse_predicate(predicate_str: str) -> Tuple[str, Optional[Union[str, int, List[int]]]]:

    if predicate_str == '@': return '@', None
    if predicate_str == '%': return '%', None
    if predicate_str == '#': return '#', None
    if predicate_str.startswith('@'):
        match = re.match(r'@(?:(-?\d+)|\(([\d\s,-]+)\))$', predicate_str)
        if match:
            if match.group(1): return '@', int(match.group(1))
            elif match.group(2):
                indices = [int(i.strip()) for i in match.group(2).split(',') if i.strip()]
                return '@', indices
    if predicate_str.startswith('!'):
        n = int(predicate_str[1:])
        if n >= 2 or n <= -1: return '!', n
    if predicate_str.startswith('$'):
        n = int(predicate_str[1:])
        if n >= 1 or n <= -1: return '$', n
    raise ValueError(f"Invalid predicate format: '{predicate_str}'")


def get_actual_indices(n_or_indices: Union[int, List[int]], list_len: int) -> Optional[List[int]]:

    actual_indices = set()
    items = n_or_indices if isinstance(n_or_indices, list) else [n_or_indices]
    for n in items:
        if n == 0: return None  
        actual_idx = n - 1 if n > 0 else list_len + n
        if 0 <= actual_idx < list_len:
            actual_indices.add(actual_idx)
        else:
            return None  
    return sorted(list(actual_indices))


def navigate(procedure: List[Dict[str, str]], cache: Dict[str, List[str]], language: str) -> Tuple[Optional[Union[List[str], str]], Optional[str], Any, str]:

    current_scope_texts = cache.get('answer', [])
    last_valid_scope_for_match = current_scope_texts
    last_predicate_type, last_predicate_param = None, None

    if not procedure:
        return last_valid_scope_for_match, None, None, ""

    for i, step in enumerate(procedure):
        level, predicate_str = step.get('level'), step.get('predicate')
        step_info = f"at step {i+1} (level='{level}', predicate='{predicate_str}')"

        try:
            predicate_type, predicate_param = parse_predicate(predicate_str)
            last_predicate_type, last_predicate_param = predicate_type, predicate_param
        except (ValueError, TypeError) as e:
            return None, None, None, f"Predicate parsing failed {step_info}: {e}"

        scope_list = current_scope_texts if isinstance(current_scope_texts, list) else [current_scope_texts]
        

        if predicate_type in ['!', '$']:

            all_segments = []
            segment_to_original_text = {}  
            
            for text_segment in scope_list:
                if isinstance(text_segment, str):
                    segments = split_text(text_segment, level, language)
                    for seg in segments:
                        all_segments.append(seg)
                        segment_to_original_text[len(all_segments) - 1] = text_segment
            
            current_len = len(all_segments)
            
            if predicate_type == '!':

                if predicate_param >= 2:
                    actual_end_index = predicate_param
                else:  
                    actual_end_index = current_len + predicate_param
                
                if not (0 <= actual_end_index <= current_len):
                    return None, None, None, f"Index out of bounds {step_info}. Global scope size is {current_len}."
                

                if actual_end_index > 0:
                    last_included_segment_idx = actual_end_index - 1
                    original_text = segment_to_original_text.get(last_included_segment_idx, "")
                    

                    last_segment = all_segments[last_included_segment_idx]
                    pos = original_text.find(last_segment)
                    
                    if pos != -1:

                        end_pos = pos + len(last_segment)
                        target_text = original_text[:end_pos]
                        current_scope_texts = [target_text]
                    else:

                        target_elements = all_segments[:actual_end_index]
                        current_scope_texts = target_elements
                else:
                    current_scope_texts = []
                
            else:  # '$'

                if predicate_param >= 1:
                    actual_start_index = predicate_param - 1
                else: 
                    actual_start_index = current_len + predicate_param + 1
                
                if not (0 <= actual_start_index <= current_len):
                    return None, None, None, f"Index out of bounds {step_info}. Global scope size is {current_len}."
                
                if actual_start_index < current_len:
                    original_text = segment_to_original_text.get(actual_start_index, "")
                    start_segment = all_segments[actual_start_index]
                    

                    pos = original_text.find(start_segment)
                    
                    if pos != -1:

                        target_text = original_text[pos:]
                        current_scope_texts = [target_text]
                    else:

                        target_elements = all_segments[actual_start_index:]
                        current_scope_texts = target_elements
                else:
                    current_scope_texts = []
            
            last_valid_scope_for_match = current_scope_texts
            
        else:

            split_segments = []
            needs_split = predicate_type in ['@']
            
            if needs_split:
                for text_segment in scope_list:
                    if isinstance(text_segment, str):
                        split_segments.extend(split_text(text_segment, level, language))
            else:
                split_segments = scope_list

            current_len = len(split_segments)

            if predicate_type == '@':
                if predicate_param is None:  # '@' (every)
                    target_elements = split_segments if i == len(procedure) - 1 else current_scope_texts
                else:  # '@N' or '@(A,B)'
                    indices_to_get = get_actual_indices(predicate_param, current_len)
                    if indices_to_get is None:
                        return None, None, None, f"Index out of bounds {step_info}. Scope size is {current_len}, requested index/indices '{predicate_param}'."
                    target_elements = [split_segments[idx] for idx in indices_to_get]
                current_scope_texts = target_elements
                last_valid_scope_for_match = current_scope_texts

            elif predicate_type in ['#', '%']:
                pass  # No scope change, handled by match()

    return last_valid_scope_for_match, last_predicate_type, last_predicate_param, ""


def compare_values(actual: int, relation: str, expected: int) -> bool:

    return {
        '=': actual == expected, '>=': actual >= expected, '<=': actual <= expected,
        '>': actual > expected, '<': actual < expected, '!=': actual != expected
    }.get(relation, False)


def match(scope: Optional[Union[List[str], str]], last_predicate_type: Optional[str], 
         last_predicate_param: Any, rule: Dict[str, Any], language: str) -> Tuple[bool, str]:

    relation, value = rule['relation'], rule['value']
    last_step = rule.get('procedure', [{}])[-1]
    level, description = last_step.get('level'), last_step.get('description')
    
    if scope is None:
        return False, "Validation failed because the navigation scope became empty or invalid before the final step."

    scope_list = scope if isinstance(scope, list) else ([scope] if scope else [])
    scope_list = [s for s in scope_list if isinstance(s, str)]

    # 1. Count Comparison (#)
    if last_predicate_type == '#':
        count = 0
        if description is not None:  # Count occurrences of a pattern/literal
            pattern_to_find = str(description)
            is_regex = (level == 'pattern')
            regex_pattern = pattern_to_find if is_regex else re.escape(pattern_to_find)
            try:
                for text_segment in scope_list:
                    count += len(re.findall(regex_pattern, text_segment, flags=re.IGNORECASE if language=='english' else 0))
            except re.error as e:
                return False, f"Invalid regex pattern '{regex_pattern}' for counting: {e}"
        elif level:  # Count elements after splitting
            for text_segment in scope_list:
                count += len(split_text(text_segment, level, language))
        else:
            return False, "Counting '#' predicate requires a 'level' or 'description'."

        is_match = compare_values(count, relation, int(value))
        if is_match:
            return True, ""
        return False, f"Count mismatch for level '{level}'. Found {count}, expected '{relation} {value}'."

    # 2. Separator Comparison (%)
    if last_predicate_type == '%':
        return True, ""  # Placeholder for separator logic

    # 3. String Content Comparison (@, !, $)
    if last_predicate_type in ['@', '!', '$']:
        actual_text = "".join(scope_list)
        expected_value = str(value)

        if relation in ['equal', 'notequal', 'contain', 'notcontain']:
            actual_text_processed = actual_text.strip()
        else:  # startswith, endswith
            actual_text_processed = actual_text

        case_sensitive = (relation == 'equal' and level in {'character', 'word', 'letter'})
        
        actual_cmp = actual_text_processed if case_sensitive else actual_text_processed.lower()
        expected_cmp = expected_value if case_sensitive else expected_value.lower()

        result = False
        if relation == 'startswith': result = actual_cmp.startswith(expected_cmp)
        elif relation == 'endswith': result = actual_cmp.endswith(expected_cmp)
        elif relation == 'equal': result = actual_cmp == expected_cmp
        elif relation == 'contain': result = expected_cmp in actual_cmp
        elif relation == 'notstartswith': result = not actual_cmp.startswith(expected_cmp)
        elif relation == 'notendswith': result = not actual_cmp.endswith(expected_cmp)
        elif relation == 'notequal': result = actual_cmp != expected_cmp
        elif relation == 'notcontain': result = expected_cmp not in actual_cmp

        if result:
            return True, ""
        
        actual_snippet = actual_text.strip()
        if len(actual_snippet) > 70:
            actual_snippet = actual_snippet[:35] + "..." + actual_snippet[-35:]
        reason = (f"String content mismatch. Relation '{relation}' failed for value '{expected_value}'. "
                  f"Actual text: '{actual_snippet}'.")
        return False, reason

    return False, f"Unsupported predicate type '{last_predicate_type}' for final matching."


def check_rule(rule: Dict[str, Any], cache: Dict[str, List[str]], language: str, recursion_depth=0) -> Tuple[bool, str]:

    procedure = rule.get('procedure', [])
    if not procedure:
        return False, "Rule is invalid: 'procedure' list is missing or empty."
    if recursion_depth > 10:
        return False, "Maximum recursion depth exceeded."


    distributive_trigger_index = -1
    for i, step in enumerate(procedure[:-1]):
        if step.get('predicate', '').startswith('@'):
            distributive_trigger_index = i
            break

    if distributive_trigger_index != -1:

        prefix_procedure = procedure[:distributive_trigger_index]
        at_step = procedure[distributive_trigger_index]
        suffix_procedure = procedure[distributive_trigger_index + 1:]

        parent_scope, _, _, nav_error = navigate(prefix_procedure, cache, language)
        if nav_error:
            return False, f"Failed navigating to distributive '@' step: {nav_error}"

        scope_list = parent_scope if isinstance(parent_scope, list) else ([parent_scope] if parent_scope else [])
        base_elements = []
        for segment in scope_list:
            base_elements.extend(split_text(segment, at_step['level'], language))

        try:
            _, pparam = parse_predicate(at_step['predicate'])
            indices_to_check = get_actual_indices(pparam, len(base_elements)) if pparam is not None else range(len(base_elements))
            if indices_to_check is None:
                 return False, f"Index out of bounds at distributive step (level='{at_step['level']}', predicate='{at_step['predicate']}')."
        except (ValueError, TypeError) as e:
            return False, f"Invalid predicate at distributive step: {e}"

        if not indices_to_check:
             return True, ""

        sub_rule = {**rule, 'procedure': suffix_procedure}
        for i, element_idx in enumerate(indices_to_check):
            element_text = base_elements[element_idx]
            element_cache = build_segmentation_cache(element_text, language)

            passed, reason = check_rule(sub_rule, element_cache, language, recursion_depth + 1)
            if not passed:
                return False, f"Distributive check failed on element {i+1} (original index {element_idx}): {reason}"
        
        return True, ""
    else:

        final_scope, last_pred_type, last_pred_param, nav_error = navigate(procedure, cache, language)
        if nav_error:
            return False, nav_error
        
        return match(final_scope, last_pred_type, last_pred_param, rule, language)
    

def generate_candidates(answer: str) -> List[Tuple[str, str]]:

    candidates = []
    

    candidates.append((answer, "identity"))
    

    def remove_markdown(text: str) -> str:

        result = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  
        result = re.sub(r'\*([^*]+)\*', r'\1', result)    
        return result
    
    def remove_first_line(text: str) -> str:

        lines = text.splitlines()
        if len(lines) > 1:
            return '\n'.join(lines[1:])
        return ""  
    
    def remove_last_line(text: str) -> str:

        lines = text.splitlines()
        if len(lines) > 1:
            return '\n'.join(lines[:-1])
        return "" 
    

    markdown_removed = remove_markdown(answer)
    if markdown_removed != answer:  
        candidates.append((markdown_removed, "remove_markdown"))
    
    first_removed = remove_first_line(answer)
    if first_removed:  
        candidates.append((first_removed, "remove_first_line"))
    
    last_removed = remove_last_line(answer)
    if last_removed:  
        candidates.append((last_removed, "remove_last_line"))
    

    if markdown_removed != answer and first_removed:
        combined = remove_first_line(markdown_removed)
        if combined:
            candidates.append((combined, "remove_markdown_and_first_line"))
    

    if markdown_removed != answer and last_removed:
        combined = remove_last_line(markdown_removed)
        if combined:
            candidates.append((combined, "remove_markdown_and_last_line"))
    

    if len(answer.splitlines()) > 2: 
        combined = remove_last_line(first_removed)
        if combined:
            candidates.append((combined, "remove_first_and_last_line"))
    

    if markdown_removed != answer and len(answer.splitlines()) > 2:
        combined = remove_first_line(remove_last_line(markdown_removed))
        if combined:
            candidates.append((combined, "remove_markdown_first_and_last_line"))
    

    seen = set()
    unique_candidates = []
    for candidate, candidate_type in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append((candidate, candidate_type))
    
    return unique_candidates


def validate_answer_with_candidates(answer: Any,
                                  instruction_pattern: List[Dict[str, Any]],
                                  language: str) -> Tuple[bool, List[Dict[str, Any]]]:
    
    reason_list = []
    any_passed = False
    

    if answer is None:
        reason_list.append({
            "candidate_type": "identity",
            "label": False,
            "reason": {
                "follow_all_list": [],
                "wrong_reason": "Answer is None"
            }
        })
        return False, reason_list
    
    if not isinstance(answer, str):
        reason_list.append({
            "candidate_type": "identity", 
            "label": False,
            "reason": {
                "follow_all_list": [],
                "wrong_reason": f"Answer must be string type, got {type(answer).__name__}"
            }
        })
        return False, reason_list
    

    candidates = generate_candidates(answer)
    

    if not candidates:
        reason_list.append({
            "candidate_type": "identity",
            "label": False,
            "reason": {
                "follow_all_list": [],
                "wrong_reason": "No valid candidates could be generated"
            }
        })
        return False, reason_list
    

    for candidate_text, candidate_type in candidates:

        passed, reason = validate_single_answer(
            candidate_text,
            instruction_pattern,
            language
        )
        
        reason_list.append({
            "candidate_type": candidate_type,
            "label": passed,
            "reason": reason
        })
        
        if passed:
            any_passed = True
    
    return any_passed, reason_list




def process_directory(input_dir: str, output_dir: str):


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return
    

    log_file_path = os.path.join(output_dir, "processing_log.txt")
    

    total_start_time = time.time()
    file_stats = []
    
    print(f"Found {len(jsonl_files)} JSONL files to process")
    print("-" * 60)
    

    for file_idx, input_file_path in enumerate(jsonl_files):
        file_name = os.path.basename(input_file_path)
        output_file_path = os.path.join(output_dir, file_name)
        
        print(f"\n[{file_idx + 1}/{len(jsonl_files)}] Processing: {file_name}")
        

        stats = process_single_file(input_file_path, output_file_path, file_idx + 1, len(jsonl_files))
        file_stats.append({
            'file_name': file_name,
            'stats': stats
        })
    

    total_time = time.time() - total_start_time
    

    write_processing_log(log_file_path, file_stats, total_time)
    
    print("\n" + "=" * 60)
    print(f"Processing completed in {total_time:.2f} seconds")
    print(f"Log saved to: {log_file_path}")


def process_single_file(input_path: str, output_path: str, file_num: int, total_files: int) -> Dict[str, Any]:

    start_time = time.time()
    

    stats = {
        'total_items': 0,
        'strict_passed': 0,
        'loose_passed': 0,
        'processing_errors': 0,
        'processing_time': 0
    }
    

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    stats['total_items'] = len(lines)
    

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for idx, line in enumerate(lines):

            if idx % 10 == 0 or idx == len(lines) - 1:
                progress = (idx + 1) / len(lines) * 100
                print(f"\r  Progress: {idx + 1}/{len(lines)} ({progress:.1f}%)", end='', flush=True)
            
            try:

                item = json.loads(line.strip())
                

                answer = item.get('answer')
                meta_info = item.get('meta_info', {})
                

                instruction_pattern_str = meta_info.get('instruction_pattern')
                language = meta_info.get('instruction_pattern_language')
                

                verify_result = {
                    'strict': {
                        'label': 0,
                        'score': '',
                        'reason': {
                            'follow_all_list': [],
                            'wrong_reason': 'Missing instruction_pattern or language'
                        }
                    },
                    'loose': {
                        'label': 0,
                        'score': '',
                        'reason': []
                    }
                }
                

                if instruction_pattern_str and language:
                    try:

                        instruction_pattern = json.loads(instruction_pattern_str)

                        strict_passed, strict_reason = validate_single_answer(
                            answer,
                            instruction_pattern,
                            language
                        )
                        
                        verify_result['strict'] = {
                            'label': 1 if strict_passed else 0,
                            'score': '',
                            'reason': strict_reason
                        }
                        
                        if strict_passed:
                            stats['strict_passed'] += 1
                        

                        loose_passed, loose_reasons = validate_answer_with_candidates(
                            answer,
                            instruction_pattern,
                            language
                        )
                        
                        verify_result['loose'] = {
                            'label': 1 if loose_passed else 0,
                            'score': '',
                            'reason': loose_reasons
                        }
                        
                        if loose_passed:
                            stats['loose_passed'] += 1
                            
                    except json.JSONDecodeError as e:
                        verify_result['strict']['reason']['wrong_reason'] = f'Invalid instruction_pattern JSON: {str(e)}'
                        verify_result['loose']['reason'] = [{
                            'candidate_type': 'identity',
                            'label': False,
                            'reason': {
                                'follow_all_list': [],
                                'wrong_reason': f'Invalid instruction_pattern JSON: {str(e)}'
                            }
                        }]
                    except Exception as e:
                        verify_result['strict']['reason']['wrong_reason'] = f'Validation error: {str(e)}'
                        verify_result['loose']['reason'] = [{
                            'candidate_type': 'identity',
                            'label': False,
                            'reason': {
                                'follow_all_list': [],
                                'wrong_reason': f'Validation error: {str(e)}'
                            }
                        }]
                        stats['processing_errors'] += 1
                

                if 'meta_info' not in item:
                    item['meta_info'] = {}
                

                item['meta_info']['verify_result'] = verify_result
                

                out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:

                out_f.write(line)
                stats['processing_errors'] += 1
            except Exception as e:

                print(f"\n  Error processing item {idx + 1}: {str(e)}")
                out_f.write(line)
                stats['processing_errors'] += 1
    

    stats['processing_time'] = time.time() - start_time
    

    print(f"\n  Completed in {stats['processing_time']:.2f}s")
    print(f"  Strict mode pass rate: {stats['strict_passed']}/{stats['total_items']} "
          f"({stats['strict_passed']/stats['total_items']*100:.1f}%)")
    print(f"  Loose mode pass rate: {stats['loose_passed']}/{stats['total_items']} "
          f"({stats['loose_passed']/stats['total_items']*100:.1f}%)")
    
    return stats


def write_processing_log(log_path: str, file_stats: List[Dict], total_time: float):

    with open(log_path, 'w', encoding='utf-8') as f:

        f.write("=" * 60 + "\n")
        f.write("PROCESSING LOG\n")
        f.write("=" * 60 + "\n")
        f.write(f"Processing Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Processing Time: {total_time:.2f} seconds\n")
        f.write("\n")
        

        total_items = sum(fs['stats']['total_items'] for fs in file_stats)
        total_strict_passed = sum(fs['stats']['strict_passed'] for fs in file_stats)
        total_loose_passed = sum(fs['stats']['loose_passed'] for fs in file_stats)
        total_errors = sum(fs['stats']['processing_errors'] for fs in file_stats)
        

        f.write("OVERALL STATISTICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total Files Processed: {len(file_stats)}\n")
        f.write(f"Total Items Processed: {total_items}\n")
        f.write(f"Total Processing Errors: {total_errors}\n")
        f.write(f"Overall Strict Mode Pass Rate: {total_strict_passed}/{total_items} "
                f"({total_strict_passed/total_items*100:.2f}%)\n")
        f.write(f"Overall Loose Mode Pass Rate: {total_loose_passed}/{total_items} "
                f"({total_loose_passed/total_items*100:.2f}%)\n")
        f.write("\n")
        

        f.write("FILE-BY-FILE STATISTICS\n")
        f.write("-" * 60 + "\n")
        
        for idx, fs in enumerate(file_stats):
            stats = fs['stats']
            f.write(f"\n[{idx + 1}] {fs['file_name']}\n")
            f.write(f"  Processing Time: {stats['processing_time']:.2f}s\n")
            f.write(f"  Total Items: {stats['total_items']}\n")
            f.write(f"  Processing Errors: {stats['processing_errors']}\n")
            f.write(f"  Strict Mode Pass Rate: {stats['strict_passed']}/{stats['total_items']} "
                    f"({stats['strict_passed']/stats['total_items']*100:.2f}%)\n")
            f.write(f"  Loose Mode Pass Rate: {stats['loose_passed']}/{stats['total_items']} "
                    f"({stats['loose_passed']/stats['total_items']*100:.2f}%)\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("END OF LOG\n")
        f.write("=" * 60 + "\n")



if __name__ == "__main__":

    test_input_dir = ""
    test_output_dir = ""
    
    

    print("Starting directory processing...")
    process_directory(test_input_dir, test_output_dir)