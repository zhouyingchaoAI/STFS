"""
自然语言查询API
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import time
import json
import asyncio
from datetime import datetime, date
from decimal import Decimal

from app.models.query import QueryRequest, QueryResponse
from app.config import settings
from app.core.nl2sql.hybrid_engine import HybridNL2SQLEngine
from app.core.query_executor.query_executor import QueryExecutor
from app.core.result_processor.formatter import ResultFormatter
from app.core.result_processor.chart_generator import ChartGenerator
from app.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)


# 初始化组件（使用混合引擎）
nl2sql_engine = HybridNL2SQLEngine()
query_executor = QueryExecutor()
result_formatter = ResultFormatter()
chart_generator = ChartGenerator()


@router.post("/query", response_model=QueryResponse)
async def natural_language_query(request: QueryRequest):
    """
    自然语言查询接口
    
    将自然语言问题转换为SQL并执行查询
    """
    start_time = time.time()
    process_steps = []  # 记录处理过程
    
    # 打印对话提问信息
    logger.info("=" * 80)
    logger.info("📝 收到对话提问")
    logger.info(f"   问题: {request.question}")
    logger.info(f"   查询选项: {request.options}")
    if request.conversation_history:
        logger.info(f"   对话历史 (共{len(request.conversation_history)}条):")
        for i, hist in enumerate(request.conversation_history, 1):
            logger.info(f"     [{i}] 问题: {hist.get('question', 'N/A')}")
            if hist.get('error'):
                logger.info(f"           错误: {hist.get('error', 'N/A')}")
            if hist.get('sql'):
                logger.info(f"           SQL: {hist.get('sql', 'N/A')[:100]}...")
    else:
        logger.info("   对话历史: 无")
    logger.info("=" * 80)
    
    try:
        # 1. NL2SQL转换
        logger.info(f"Processing query: {request.question}")
        query_options = request.options or {}
        use_llm = query_options.get("use_llm", False)  # 是否强制使用LLM
        
        process_steps.append({
            "step": "理解问题",
            "status": "processing",
            "message": "正在分析自然语言问题...",
            "timestamp": time.time()
        })
        
        nl2sql_start = time.time()
        # 获取对话历史（用于多轮对话修正）
        conversation_history = request.conversation_history if request.conversation_history is not None else []
        sql_result = nl2sql_engine.convert(request.question, use_llm=use_llm, conversation_history=conversation_history)
        nl2sql_time = time.time() - nl2sql_start
        
        if not sql_result or not sql_result.get("sql"):
            process_steps.append({
                "step": "理解问题",
                "status": "error",
                "message": "无法理解查询意图",
                "timestamp": time.time()
            })
            raise HTTPException(
                status_code=400,
                detail={
                    "code": 400,
                    "message": "无法理解查询意图，请尝试重新表述问题",
                    "data": None
                }
            )
        
        sql_query = sql_result["sql"]
        intent = sql_result.get("intent")
        entities = sql_result.get("entities", {})
        engine_type = sql_result.get("engine_type", "rule")  # rule 或 llm
        thinking_process = sql_result.get("thinking_process", "")
        
        process_steps.append({
            "step": "理解问题",
            "status": "success",
            "message": f"使用{engine_type}引擎完成意图识别",
            "details": {
                "intent": intent,
                "entities": entities,
                "thinking": thinking_process if thinking_process else f"使用{engine_type}引擎分析问题：\n1. 识别查询意图为：{intent}\n2. 提取实体信息：{entities}\n3. 匹配查询模板并生成SQL"
            },
            "duration": round(nl2sql_time, 3),
            "timestamp": time.time()
        })
        
        logger.info(f"Generated SQL: {sql_query}")
        logger.info(f"Intent: {intent}, Entities: {entities}")
        
        # 2. 生成SQL
        process_steps.append({
            "step": "生成SQL",
            "status": "processing",
            "message": "正在生成SQL查询语句...",
            "timestamp": time.time()
        })
        
        process_steps.append({
            "step": "生成SQL",
            "status": "success",
            "message": "SQL语句生成完成",
            "details": {
                "sql": sql_query
            },
            "timestamp": time.time()
        })
        
        # 3. 执行查询
        process_steps.append({
            "step": "执行查询",
            "status": "processing",
            "message": "正在执行数据库查询...",
            "timestamp": time.time()
        })
        
        database = query_options.get("database", "master")
        max_rows = query_options.get("max_rows", 10000)
        
        query_start = time.time()
        query_result = query_executor.execute(
            sql=sql_query,
            database=database,
            max_rows=max_rows
        )
        query_time = time.time() - query_start
        
        process_steps.append({
            "step": "执行查询",
            "status": "success",
            "message": f"查询执行完成，返回 {query_result.get('row_count', 0)} 行数据",
            "details": {
                "row_count": query_result.get("row_count", 0),
                "database": database
            },
            "duration": round(query_time, 3),
            "timestamp": time.time()
        })
        
        # 4. 格式化结果
        process_steps.append({
            "step": "处理结果",
            "status": "processing",
            "message": "正在格式化查询结果...",
            "timestamp": time.time()
        })
        
        format_start = time.time()
        formatted_result = result_formatter.format(query_result)
        format_time = time.time() - format_start
        
        # 5. 生成图表配置
        chart_start = time.time()
        chart_config = chart_generator.generate(
            data=formatted_result,
            intent=intent
        )
        chart_time = time.time() - chart_start
        
        process_steps.append({
            "step": "处理结果",
            "status": "success",
            "message": "结果格式化和图表配置完成",
            "details": {
                "has_chart": chart_config is not None
            },
            "duration": round(format_time + chart_time, 3),
            "timestamp": time.time()
        })
        
        execution_time = time.time() - start_time
        
        # 6. 构建响应
        response_data = {
            "sql": sql_query,
            "result": formatted_result,
            "chart_config": chart_config,
            "execution_time": round(execution_time, 3),
            "row_count": len(formatted_result),
            "process_steps": process_steps  # 添加过程信息
        }
        
        return QueryResponse(
            code=200,
            message="success",
            data=response_data,
            metadata={
                "intent": intent,
                "entities": entities,
                "engine_type": engine_type,
                "thinking_process": thinking_process
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query execution error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "code": 500,
                "message": f"查询执行失败: {str(e)}",
                "data": None
            }
        )


@router.post("/query/stream")
async def natural_language_query_stream(request: QueryRequest):
    """
    自然语言查询接口（流式版本）
    
    使用Server-Sent Events (SSE)实时返回思考过程和查询结果
    """
    # 打印对话提问信息
    logger.info("=" * 80)
    logger.info("📝 收到对话提问 (流式)")
    logger.info(f"   问题: {request.question}")
    logger.info(f"   查询选项: {request.options}")
    if request.conversation_history:
        logger.info(f"   对话历史 (共{len(request.conversation_history)}条):")
        for i, hist in enumerate(request.conversation_history, 1):
            logger.info(f"     [{i}] 问题: {hist.get('question', 'N/A')}")
            if hist.get('error'):
                logger.info(f"           错误: {hist.get('error', 'N/A')}")
            if hist.get('sql'):
                logger.info(f"           SQL: {hist.get('sql', 'N/A')[:100]}...")
    else:
        logger.info("   对话历史: 无")
    logger.info("=" * 80)
    
    async def generate():
        start_time = time.time()
        process_steps = []
        
        try:
            query_options = request.options or {}
            use_llm = query_options.get("use_llm", False)
            
            # 发送开始思考的信号
            yield f"data: {json.dumps({'type': 'thinking_start', 'message': '开始分析问题...'}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.001)  # 确保立即发送
            
            # 1. NL2SQL转换（流式）
            if use_llm:
                # 使用LLM引擎的流式调用
                from app.core.nl2sql.llm_based_engine import LLMBasedNL2SQLEngine
                llm_engine = LLMBasedNL2SQLEngine()
                
                # 获取对话历史（用于多轮对话修正）
                conversation_history = request.conversation_history if request.conversation_history is not None else []
                
                # 流式调用LLM
                full_response = ""
                
                prompt = llm_engine._build_prompt(request.question, conversation_history)
                logger.info(f"开始流式调用LLM，prompt长度: {len(prompt)}")
                
                try:
                    response_stream = llm_engine._call_ollama_stream(prompt)
                    logger.info("Ollama流式响应已启动")
                except Exception as stream_error:
                    logger.error(f"启动Ollama流式响应失败: {stream_error}", exc_info=True)
                    # 发送错误信息
                    error_msg = f"无法连接到LLM服务: {str(stream_error)}"
                    error_content = f"❌ {error_msg}\n"
                    yield f"data: {json.dumps({'type': 'thinking', 'content': error_content}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'error', 'message': error_msg, 'details': str(stream_error)}, ensure_ascii=False)}\n\n"
                    return
                
                # 用于跟踪是否已经遇到SELECT
                found_select = False
                
                chunk_count = 0
                thinking_char_count = 0
                has_received_any_chunk = False
                
                # 用于跟踪是否已经发送了初始提示（如果等待一段时间还没收到内容，再发送提示）
                initial_prompt_sent = False
                first_chunk_time = None
                last_chunk_time = None
                no_chunk_timeout = 5.0  # 5秒没有收到chunk则发送提示
                
                logger.info("开始接收Ollama流式响应chunks...")
                
                for chunk in response_stream:
                    chunk_count += 1
                    current_time = asyncio.get_event_loop().time()
                    
                    # 检查是否长时间没有收到chunk
                    if last_chunk_time is not None and (current_time - last_chunk_time) > no_chunk_timeout:
                        logger.warning(f"超过{no_chunk_timeout}秒没有收到chunk，可能Ollama响应较慢")
                        if not has_received_any_chunk:
                            # 如果还没收到任何chunk，发送等待提示
                            # f-string中不能包含反斜杠，先定义字符串
                            waiting_msg = '⏳ 正在等待LLM响应，请稍候...\n\n'
                            yield f"data: {json.dumps({'type': 'thinking', 'content': waiting_msg}, ensure_ascii=False)}\n\n"
                    
                    if chunk:
                        # Ollama返回格式: {"response": "text", "done": False}
                        chunk_text = chunk.get('response', '') if isinstance(chunk, dict) else str(chunk)
                        is_done = chunk.get('done', False) if isinstance(chunk, dict) else False
                        
                        # 更新最后收到chunk的时间
                        last_chunk_time = current_time
                        
                        if chunk_text:
                            has_received_any_chunk = True
                            
                            # 记录第一个chunk的时间
                            if first_chunk_time is None:
                                first_chunk_time = current_time
                            
                            full_response += chunk_text
                            
                            # 检查是否包含SELECT（表示SQL开始）
                            if 'SELECT' in chunk_text.upper() and not found_select:
                                # 如果chunk中包含SELECT，提取SELECT之前的部分作为思考内容
                                select_index = chunk_text.upper().find('SELECT')
                                if select_index > 0:
                                    thinking_part = chunk_text[:select_index]
                                    # 立即发送思考内容（整个部分）
                                    if thinking_part:
                                        yield f"data: {json.dumps({'type': 'thinking', 'content': thinking_part}, ensure_ascii=False)}\n\n"
                                        thinking_char_count += len(thinking_part)
                                
                                # 标记已找到SELECT，之后的内容不再作为思考内容发送
                                found_select = True
                                # 继续收集SELECT及之后的内容到full_response（不发送，但需要完整响应来解析SQL）
                                
                            elif not found_select:
                                # 没有SELECT，立即发送这个chunk作为思考内容
                                # 直接发送整个chunk（Ollama的chunk通常已经很小了，每个chunk可能只有几个字符）
                                # 即使chunk很小（如'<think>'、'\n'等），也要发送
                                if chunk_text:
                                    # 如果这是第一个有内容的chunk，且之前没有发送过初始提示，可以添加一个简单的提示
                                    if thinking_char_count == 0 and not initial_prompt_sent:
                                        # 如果有对话历史，添加修正提示
                                        if conversation_history:
                                            last_error = conversation_history[-1].get("error")
                                            if last_error:
                                                # f-string中不能包含反斜杠，先定义字符串
                                                correction_msg = '📝 检测到之前的查询错误，正在根据错误信息进行修正...\n\n'
                                                yield f"data: {json.dumps({'type': 'thinking', 'content': correction_msg}, ensure_ascii=False)}\n\n"
                                                initial_prompt_sent = True
                                    
                                    # 立即发送，不等待，不缓冲，确保每个token都立即显示
                                    # 使用flush确保立即发送（虽然SSE会自动flush，但这里明确说明）
                                    thinking_event = f"data: {json.dumps({'type': 'thinking', 'content': chunk_text}, ensure_ascii=False)}\n\n"
                                    yield thinking_event
                                    thinking_char_count += len(chunk_text)
                            
                            # 如果已经找到SELECT，继续收集剩余内容到full_response（不发送，但需要完整响应）
                        
                        # 如果done=True，表示流式响应结束
                        if is_done:
                            # 思考过程完成，立即发送完成信号
                            if thinking_char_count > 0:
                                yield f"data: {json.dumps({'type': 'thinking_complete', 'message': '思考过程已完成'}, ensure_ascii=False)}\n\n"
                                await asyncio.sleep(0.001)
                            break
                
                # 检查是否收到了任何chunk
                if not has_received_any_chunk:
                    logger.warning("Ollama流式响应没有返回任何chunk！")
                    warning_content = "⚠️ LLM没有返回任何响应，请检查Ollama服务是否正常运行\n"
                    yield f"data: {json.dumps({'type': 'thinking', 'content': warning_content}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'error', 'message': 'LLM没有返回响应', 'details': 'Ollama流式响应为空'}, ensure_ascii=False)}\n\n"
                    return
                
                # 如果收到了chunk但没有发送任何思考内容（可能直接返回了SQL），发送一个提示
                if has_received_any_chunk and thinking_char_count == 0 and not found_select:
                    # 检查full_response是否直接是SQL（没有思考过程）
                    if 'SELECT' in full_response.upper():
                        # 直接返回了SQL，没有思考过程，发送一个简短提示
                        info_msg = "💭 正在分析问题并生成SQL查询...\n\n"
                        yield f"data: {json.dumps({'type': 'thinking', 'content': info_msg}, ensure_ascii=False)}\n\n"
                        thinking_char_count += len(info_msg)
                
                # 记录完整响应用于调试
                # 流式处理完成
                has_select = 'SELECT' in full_response.upper()
                resp_preview = full_response[:200] if len(full_response) > 200 else full_response
                logger.info(f"LLM完整响应长度: {len(full_response)}, 包含SELECT: {has_select}")
                logger.info(f"LLM响应前200字符: {resp_preview}")
                if not has_select:
                    logger.warning(f"LLM响应中未找到SELECT语句，完整响应: {full_response[:500]}")
                
                # 解析完整响应
                try:
                    sql_result = llm_engine._parse_response(full_response, request.question)
                except Exception as parse_error:
                    logger.error(f"LLM响应解析失败: {parse_error}")
                    logger.error(f"完整响应内容: {full_response[:1000]}")
                    # 发送解析错误信息
                    parse_error_msg = f"\n\n[LLM响应解析失败: {str(parse_error)}]"
                    yield f"data: {json.dumps({'type': 'thinking', 'content': parse_error_msg}, ensure_ascii=False)}\n\n"
                    sql_result = None
                
                # 如果LLM解析失败，回退到规则引擎
                if not sql_result or not sql_result.get("sql"):
                    logger.warning("LLM解析失败，回退到规则引擎")
                    resp_preview = full_response[:500] if len(full_response) > 500 else full_response
                    logger.warning(f"完整响应内容（前500字符）: {resp_preview}")
                    # 发送失败信息和完整响应用于调试
                    fallback_msg = f"\n\n[LLM解析失败，切换到规则引擎...]\n[已收集的LLM响应（前500字符）:\n{resp_preview}]"
                    yield f"data: {json.dumps({'type': 'thinking', 'content': fallback_msg}, ensure_ascii=False)}\n\n"
                    conversation_history = request.conversation_history if request.conversation_history is not None else []
                    sql_result = nl2sql_engine.convert(request.question, use_llm=False, conversation_history=conversation_history)
            else:
                # 使用规则引擎（非流式）
                yield f"data: {json.dumps({'type': 'thinking', 'content': '正在使用规则引擎分析问题...'}, ensure_ascii=False)}\n\n"
                sql_result = nl2sql_engine.convert(request.question, use_llm=False)
            
            if not sql_result or not sql_result.get("sql"):
                yield f"data: {json.dumps({'type': 'error', 'message': '无法理解查询意图'}, ensure_ascii=False)}\n\n"
                return
            
            sql_query = sql_result["sql"]
            thinking_process = sql_result.get("thinking_process", "")
            
            # 1. 立即发送SQL生成完成（阶段1完成）
            yield f"data: {json.dumps({'type': 'sql_generated', 'sql': sql_query, 'thinking': thinking_process}, ensure_ascii=False)}\n\n"
            logger.info("[阶段1完成] SQL生成完成，立即发送到前端")
            await asyncio.sleep(0.001)  # 确保立即发送
            
            # 2. 执行查询（阶段2）- 多轮对话模式：失败时记录错误，不自动重试
            database = query_options.get("database", "master")
            max_rows = query_options.get("max_rows", 10000)
            
            yield f"data: {json.dumps({'type': 'step', 'step': '执行查询', 'status': 'processing', 'message': '正在执行数据库查询...'}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.001)  # 确保立即发送
            
            try:
                query_result = query_executor.execute(
                    sql=sql_query,
                    database=database,
                    max_rows=max_rows
                )
            except Exception as db_error:
                # 查询失败，记录错误信息到对话历史，不自动重试
                error_msg = str(db_error)
                logger.error(f"数据库查询失败: {error_msg}")
                logger.error(f"失败的SQL: {sql_query}")
                
                # 确保思考过程已经完整发送（如果还没有发送完成信号，现在发送）
                if thinking_process:
                    # 发送思考过程完成信号（如果之前没有发送）
                    yield f"data: {json.dumps({'type': 'thinking_complete', 'message': '思考过程已完成'}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.001)
                
                # 发送错误信息到前端（包含错误详情和失败的SQL，供下一轮对话使用）
                yield f"data: {json.dumps({'type': 'step', 'step': '执行查询', 'status': 'error', 'message': f'查询执行失败: {error_msg}'}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.001)
                
                # 发送详细的错误信息，包含失败的SQL和思考过程，供下一轮对话修正使用
                error_details = {
                    "error": error_msg,
                    "failed_sql": sql_query,
                    "original_question": request.question,
                    "thinking_process": thinking_process,  # 包含思考过程
                    "suggestion": "您可以在下一轮对话中提供更多信息，或直接说'修正SQL'、'重新查询'等，系统会根据错误信息自动修正。"
                }
                yield f"data: {json.dumps({'type': 'error', 'message': f'数据库查询失败: {error_msg}', 'details': error_details, 'sql': sql_query, 'thinking': thinking_process, 'can_retry': True}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.001)
                return
            
            # 查询执行成功
            row_count = query_result.get("row_count", 0)
            query_time = time.time() - start_time
            
            # 2.1 立即发送查询执行完成（阶段2完成）
            success_msg = f'查询执行完成，返回 {row_count} 行数据'
            yield f"data: {json.dumps({'type': 'step', 'step': '执行查询', 'status': 'success', 'message': success_msg, 'row_count': row_count}, ensure_ascii=False)}\n\n"
            logger.info(f"[阶段2完成] 查询执行完成，返回 {row_count} 行数据，立即发送到前端")
            await asyncio.sleep(0.001)  # 确保立即发送
            
            # 2.2 立即发送查询结果（部分数据）- 执行完成后立即显示
            if query_result.get("data"):
                # 发送前几行数据，让用户先看到结果
                preview_data = query_result["data"][:10]  # 前10行
                # 需要处理Decimal序列化
                def preview_serializer(obj):
                    if isinstance(obj, (datetime, date)):
                        return obj.isoformat()
                    elif isinstance(obj, Decimal):
                        if obj % 1 == 0:
                            return int(obj)
                        else:
                            return float(obj)
                    return obj
                
                # 序列化预览数据
                import json as json_lib
                preview_data_serialized = json_lib.loads(json_lib.dumps(preview_data, default=preview_serializer))
                yield f"data: {json.dumps({'type': 'result_preview', 'data': preview_data_serialized, 'total_rows': row_count, 'preview_count': len(preview_data)}, ensure_ascii=False)}\n\n"
                logger.info(f"[阶段2部分] 立即发送前10行预览数据到前端")
                await asyncio.sleep(0.001)  # 确保立即发送
            
            # 3. 格式化结果（阶段3）- 立即开始处理
            yield f"data: {json.dumps({'type': 'step', 'step': '处理结果', 'status': 'processing', 'message': '正在格式化查询结果...'}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.001)  # 确保立即发送
            
            format_start = time.time()
            formatted_result = result_formatter.format(query_result)
            format_time = time.time() - format_start
            
            # 3.1 立即发送格式化完成
            yield f"data: {json.dumps({'type': 'step', 'step': '处理结果', 'status': 'success', 'message': f'结果格式化完成，共 {row_count} 行', 'duration': round(format_time, 3)}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.001)
            
            # 3.1 立即发送格式化结果完成（阶段3完成）
            # 需要处理Decimal序列化
            def format_serializer(obj):
                if isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                elif isinstance(obj, Decimal):
                    if obj % 1 == 0:
                        return int(obj)
                    else:
                        return float(obj)
                return obj
            
            formatted_result_serialized = json.loads(json.dumps(formatted_result, default=format_serializer))
            yield f"data: {json.dumps({'type': 'result_formatted', 'data': formatted_result_serialized, 'row_count': len(formatted_result)}, ensure_ascii=False)}\n\n"
            logger.info(f"[阶段3完成] 结果格式化完成，立即发送到前端")
            await asyncio.sleep(0.001)  # 确保立即发送
            
            # 4. 生成图表配置（阶段4）
            yield f"data: {json.dumps({'type': 'step', 'step': '生成图表', 'status': 'processing', 'message': '正在生成图表配置...'}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.001)  # 确保立即发送
            
            chart_config = chart_generator.generate(
                data=formatted_result,
                intent=sql_result.get("intent")
            )
            
            # 4.1 立即发送图表配置完成（阶段4完成）
            yield f"data: {json.dumps({'type': 'chart_generated', 'chart_config': chart_config}, ensure_ascii=False)}\n\n"
            logger.info(f"[阶段4完成] 图表配置生成完成，立即发送到前端")
            await asyncio.sleep(0.001)  # 确保立即发送
            
            execution_time = time.time() - start_time
            
            # 5. 发送最终完成信号（需要处理datetime和Decimal序列化）
            def json_serializer(obj):
                """自定义JSON序列化器，处理datetime和Decimal对象"""
                if isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                elif isinstance(obj, Decimal):
                    # Decimal转换为float或int
                    if obj % 1 == 0:
                        return int(obj)
                    else:
                        return float(obj)
                raise TypeError(f"Type {type(obj)} not serializable")
            
            final_data = {
                "type": "complete",
                "data": {
                    "sql": sql_query,
                    "result": formatted_result,
                    "chart_config": chart_config,
                    "execution_time": round(execution_time, 3),
                    "row_count": len(formatted_result)
                },
                "metadata": {
                    "intent": sql_result.get("intent"),
                    "entities": sql_result.get("entities", {}),
                    "engine_type": "llm" if use_llm else "rule",
                    "thinking_process": thinking_process
                }
            }
            
            yield f"data: {json.dumps(final_data, default=json_serializer, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            logger.error(f"Stream query error: {e}", exc_info=True)
            
            # 即使失败，也要发送已收集的思维过程和完整响应
            if 'full_response' in locals() and full_response:
                logger.info(f"发送失败时的完整响应用于调试，长度: {len(full_response)}")
                resp_preview = full_response[:500] if len(full_response) > 500 else full_response
                error_info = f"\n\n[❌ 错误发生]\n已收集的响应内容（前500字符）：\n{resp_preview}..."
                yield f"data: {json.dumps({'type': 'thinking', 'content': error_info}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.001)
            
            # 立即发送错误信息到对话中
            error_msg = f'查询执行失败: {str(e)}'
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg, 'details': str(e)}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.001)
            
            # 发送错误步骤
            yield f"data: {json.dumps({'type': 'step', 'step': '执行失败', 'status': 'error', 'message': error_msg}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.001)
            
            # 发送完整的错误堆栈用于调试
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"完整错误堆栈:\n{error_trace}")
            yield f"data: {json.dumps({'type': 'error_detail', 'traceback': error_trace}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

