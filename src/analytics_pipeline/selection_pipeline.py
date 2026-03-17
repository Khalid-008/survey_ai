import os



import sys



import warnings



from typing import Any



import pandas as pd




# ─── path setup ───────────────────────────────────────────────────────────────



current_file_path = os.path.abspath(__file__)



src_dir = os.path.dirname(os.path.dirname(current_file_path))



if src_dir not in sys.path:



    sys.path.insert(0, src_dir)




from llms.models import model, qwen3_model



from data.operations import get_selection_questions_data, execute_raw_query



from analytics_pipeline.util.selection_utils import (



    extract_sql,



    clean_sql_queries,



    validate_query,



    format_questions_block,



    save_results_to_file,
)



from agents.prompt.selection_analytics_prompt import selection_analytics_prompt




warnings.filterwarnings("ignore")




# ============================================================================



# STEP 1 — FETCH & GROUP DATA



# ============================================================================



def group_answers_by_question(df: pd.DataFrame) -> dict[int, dict[str, Any]]:



    grouped: dict[int, dict[str, Any]] = {}



    # Use answer_label when available (resolved text), fall back to raw answer


    answer_col = "answer_label" if "answer_label" in df.columns else "answer"



    for question_id, group in df.groupby("question_id"):



        grouped[int(question_id)] = {



            "text":    group["question_ar"].iloc[0],



            "type":    group["question_type"].iloc[0],



            "answers": group[answer_col].dropna().tolist(),



        }



    return grouped





def get_distinct_answers_per_question(grouped: dict[int, dict[str, Any]]) -> dict[int, list[str]]:



    distinct: dict[int, list[str]] = {}



    for qid, info in grouped.items():



        unique_answers = sorted(set(str(a) for a in info["answers"] if a))



        distinct[qid] = unique_answers



        print(f"   Q{qid} ({info['text'][:40]}...): {len(unique_answers)} distinct answers")
    return distinct





def get_sample_answers_per_question(df: pd.DataFrame, n: int = 3) -> dict[int, list[str]]:



    """Returns up to `n` full row examples (all columns) per question for agent context."""



    samples: dict[int, list[str]] = {}



    for qid, group in df.groupby("question_id"):



        rows = group.head(n).to_dict(orient="records")



        samples[int(qid)] = [str(row) for row in rows]


    return samples





# ============================================================================



# MAIN PIPELINE



# ============================================================================



def generate_sql_analisys_queries(survey_id: str, prepared: dict[str, Any], analytical_questions: str = "") -> dict[str, Any]:

    print(f"\n{'#'*60}")


    print(f"# SELECTION PIPELINE — Survey: {survey_id}")


    print(f"{'#'*60}\n")




    result: dict[str, Any] = {

        "query_results":         [],

        "errors":                [],

    }


    # قراءة questions_block من النود السابق

    questions_block = prepared.get("questions_block", "")


    if not questions_block:

        result["errors"].append("No selection questions found.")
        return result



    # 3. Generate



    print(f"\n{'─'*60}")



    print(f"🤖 Step 3 — Generating 5 analytical queries...")



    print(f"{'─'*60}")




    try:


        prompt_messages = selection_analytics_prompt.invoke({
            "questions_block":     questions_block,
            "analytical_questions": analytical_questions,
        })


        response   = qwen3_model.invoke(prompt_messages)


        raw_output = response.content


        if not raw_output:

            print(f"   ⚠️ Empty content! Full response debug:")

            print(f"      type: {type(response)}")

            print(f"      additional_kwargs: {response.additional_kwargs}")

            print(f"      response_metadata: {response.response_metadata}")


        print(f"   ✅ LLM responded ({len(raw_output)} chars).")



    except Exception as e:



        err = f"Generation failed: {e}"



        print(f"   ❌ {err}")



        result["errors"].append(err)
        return result




    # 4. Clean & split into individual queries



    queries = clean_sql_queries(raw_output)



    print(f"   📋 {len(queries)} queries extracted after cleaning.")



    print(f"\n{'═'*60}")

    print("📄 ALL CLEANED QUERIES (before execution):"
)
    print(f"{'═'*60}")

    for i, sql in enumerate(queries, start=1):

        print(f"\n── Query {i} ──\n{sql}")

    print(f"\n{'═'*60}\n")



    # 5. Validate → Execute → Store (one query at a time)



    for i, sql in enumerate(queries, start=1):



        label = f"Query {i}"



        entry: dict[str, Any] = {"label": label, "sql": sql, "result": pd.DataFrame()}




        print(f"\n{'─'*60}")



        print(f"⚡ {label}")



        print(f"{'─'*60}")



        print(sql)




        # Validate



        is_valid, issues = validate_query(sql)



        if not is_valid:



            print(f"   ⚠️  Validation warnings (auto-fixed in clean step):")



            for issue in issues:



                print(f"      {issue}")




        # Execute



        try:



            df_result      = execute_raw_query(sql)



            entry["result"] = df_result



            print(f"   ✅ {len(df_result)} rows returned.")



        except Exception as e:



            err = f"[{label}] Execution failed: {e}"



            print(f"   ❌ {err}")



            result["errors"].append(err)




        # Store



        result["query_results"].append(entry)



        print(f"   💾 Stored: {label}")




    # Summary



    total_rows = sum(len(qr["result"]) for qr in result["query_results"])



    print(f"\n{'='*60}")



    print(f"✅ DONE — Survey: {survey_id}")


    print(f"   Queries   : {len(result['query_results'])}")



    print(f"   Rows      : {total_rows}")



    if result["errors"]:



        print(f"   ⚠️  Errors ({len(result['errors'])}):")



        for err in result["errors"]:



            print(f"      • {err}")



    print(f"{'='*60}\n")


    # Save


    result["report_path"] = save_results_to_file(

        survey_number=survey_id,

        query_results=result["query_results"],

        errors=result["errors"],
    )

    return result



