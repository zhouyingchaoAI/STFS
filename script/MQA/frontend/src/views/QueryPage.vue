<template>
  <div class="chat-page">
    <!-- 对话区域 -->
    <div class="chat-container">
      <div class="chat-messages" ref="messagesContainer" @scroll="handleScroll">
        <!-- 欢迎消息 -->
        <div v-if="conversations.length === 0" class="welcome-message">
          <div class="welcome-content">
            <el-icon :size="48" class="welcome-icon"><ChatDotRound /></el-icon>
            <h2>欢迎使用地铁客流智能问数系统</h2>
            <p>我可以帮您查询地铁客流数据，请告诉我您想了解什么？</p>
            <div class="quick-suggestions">
              <div class="suggestion-title">试试这些问题：</div>
              <el-tag
                v-for="(query, index) in quickQueries"
                :key="index"
                @click="sendMessage(query)"
                class="suggestion-tag"
                effect="plain"
              >
                {{ query }}
              </el-tag>
            </div>
          </div>
        </div>

        <!-- 对话消息列表 -->
        <div
          v-for="(conv, index) in conversations"
          :key="index"
          class="conversation-item"
        >
          <!-- 用户消息 -->
          <div class="message user-message">
            <div class="message-avatar user-avatar">
              <el-icon><User /></el-icon>
            </div>
            <div class="message-content user-content">
              <div class="message-text">
                <span v-if="conv.currentTime" class="time-prefix">[{{ conv.currentTime }}]</span>
                {{ conv.originalQuestion || conv.question }}
              </div>
              <div class="message-time">{{ formatTime(conv.timestamp) }}</div>
            </div>
          </div>

              <!-- AI回复 -->
          <div class="message ai-message" v-if="conv.response || conv.loading">
            <div class="message-avatar ai-avatar">
              <el-icon><Robot /></el-icon>
            </div>
            <div class="message-content ai-content">
              <!-- 加载状态 -->
              <div v-if="conv.loading" class="loading-indicator">
                <el-icon class="is-loading"><Loading /></el-icon>
                <span>正在思考中...</span>
              </div>

              <!-- 思维链展示（Dify风格 - 树状结构） -->
              <!-- 只要有loading或thinkingProcess不为空，就显示思考容器 -->
              <div v-if="conv.loading || (conv.thinkingProcess && conv.thinkingProcess.length > 0)" class="thinking-chain-container">
                <!-- 主思考节点 -->
                <div class="thinking-node thinking-node-main">
                  <div class="thinking-node-header" @click="toggleThinkingNode(index)">
                    <div class="thinking-node-left">
                      <el-icon class="thinking-node-icon" :class="{ 'is-expanded': conv.thinkingExpanded !== false }">
                        <ArrowRight v-if="!conv.thinkingExpanded" />
                        <ArrowDown v-else />
                      </el-icon>
                      <el-icon class="thinking-node-status">
                        <Loading v-if="conv.loading" class="thinking-loading-icon" />
                        <Lightning v-else />
                      </el-icon>
                      <span class="thinking-node-title">思考过程</span>
                      <el-tag v-if="conv.loading" size="small" type="info" effect="plain" style="margin-left: 8px;">
                        思考中...
                      </el-tag>
                    </div>
                    <div class="thinking-node-right">
                      <span v-if="conv.thinkingProcess && conv.thinkingProcess.length > 0" class="thinking-node-count">
                        {{ conv.thinkingProcess.length }} 字符
                      </span>
                    </div>
                  </div>
                  <!-- 思考内容（可折叠） -->
                  <div v-show="conv.thinkingExpanded !== false" class="thinking-node-content">
                    <div class="thinking-content-text" :class="{ 'has-content': conv.thinkingProcess && conv.thinkingProcess.length > 0 }">
                      <!-- 直接显示完整文本，支持逐字符流式显示 -->
                      <!-- 使用v-html支持格式化，但确保实时更新 -->
                      <div class="thinking-text-display" v-html="formatThinkingText(conv.thinkingProcess || '')" :key="conv.thinkingProcess?.length || 0"></div>
                      <!-- 如果完全没有内容，显示占位符 -->
                      <div v-if="!conv.thinkingProcess || conv.thinkingProcess.length === 0" class="thinking-placeholder">
                        <el-icon class="thinking-placeholder-icon"><Loading /></el-icon>
                        <span>正在思考中，请稍候...</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <!-- 执行步骤展示（Dify风格 - 树状结构） -->
              <div v-if="conv.processSteps && conv.processSteps.length > 0" class="process-steps-tree">
                <div
                  v-for="(step, stepIndex) in conv.processSteps"
                  :key="stepIndex"
                  class="process-step-node"
                  :class="['step-' + step.status, { 'step-expanded': step.expanded !== false }]"
                >
                  <div class="step-node-header" @click="toggleStepNode(index, stepIndex)">
                    <div class="step-node-left">
                      <el-icon class="step-expand-icon" :class="{ 'is-expanded': step.expanded !== false }">
                        <ArrowRight v-if="!step.expanded" />
                        <ArrowDown v-else />
                      </el-icon>
                      <el-icon class="step-status-icon" :class="'status-' + step.status">
                        <Loading v-if="step.status === 'processing'" />
                        <CircleCheck v-else-if="step.status === 'success'" />
                        <CircleClose v-else-if="step.status === 'error'" />
                        <Warning v-else />
                      </el-icon>
                      <span class="step-node-title">{{ step.step }}</span>
                      <el-tag 
                        v-if="step.status === 'processing'" 
                        size="small" 
                        type="info" 
                        effect="plain"
                        style="margin-left: 8px;"
                      >
                        执行中
                      </el-tag>
                    </div>
                    <div class="step-node-right">
                      <span v-if="step.duration" class="step-duration">{{ step.duration }}s</span>
                      <span class="step-timestamp">{{ formatStepTime(step.timestamp) }}</span>
                    </div>
                  </div>
                  <div v-show="step.expanded !== false" class="step-node-content">
                    <div class="step-message">{{ step.message }}</div>
                  
                    <!-- 详细信息 -->
                    <el-collapse class="step-details-collapse">
                      <el-collapse-item title="查看详情" :name="stepIndex">
                        <!-- 错误信息 - 如果有错误优先显示 -->
                        <div v-if="step.details?.error" class="error-box">
                          <div class="error-title">❌ 错误信息</div>
                          <div class="error-content">{{ step.details.error }}</div>
                          <div v-if="step.details.errorDetails" class="error-details">
                            <div class="error-details-title">错误详情：</div>
                            <pre class="error-traceback">{{ step.details.errorDetails }}</pre>
                          </div>
                          <div v-if="step.details.traceback" class="error-traceback-box">
                            <div class="error-traceback-title">错误堆栈：</div>
                            <pre class="error-traceback">{{ step.details.traceback }}</pre>
                          </div>
                        </div>
                        
                        <!-- 思考过程 - 优先显示 -->
                        <div v-if="step.details?.thinking" class="thinking-box">
                          <div class="thinking-title">💭 思考过程</div>
                          <div class="thinking-content">{{ step.details.thinking }}</div>
                        </div>
                        
                        <div v-if="step.details?.intent" class="detail-item">
                          <strong>意图:</strong> {{ step.details.intent }}
                        </div>
                        <div v-if="step.details?.entities" class="detail-item">
                          <strong>实体:</strong>
                          <pre>{{ JSON.stringify(step.details.entities, null, 2) }}</pre>
                        </div>
                        <div v-if="step.details?.sql" class="detail-item">
                          <strong>SQL:</strong>
                          <pre class="sql-code">{{ step.details.sql }}</pre>
                        </div>
                      </el-collapse-item>
                    </el-collapse>
                  </div>
                </div>
              </div>

              <!-- 查询结果（分阶段显示） -->
              <div v-if="conv.result" class="query-result">
                <!-- SQL预览 - 可折叠，默认折叠 -->
                <el-collapse v-if="conv.result && conv.result.sql" class="result-collapse">
                  <el-collapse-item 
                    :name="`sql-${index}`"
                    :title="'生成的SQL语句'"
                  >
                    <template #title>
                      <div class="sql-header-collapse">
                        <el-icon><Document /></el-icon>
                        <span>生成的SQL语句</span>
                        <el-button 
                          text 
                          size="small" 
                          @click.stop="copySQL(conv.result.sql)"
                          style="margin-left: auto;"
                        >
                          <el-icon><DocumentCopy /></el-icon>
                          复制SQL
                        </el-button>
                      </div>
                    </template>
                    <pre class="sql-code-display">{{ conv.result.sql }}</pre>
                  </el-collapse-item>
                </el-collapse>
                
                <!-- 结果预览提示 -->
                <div v-if="conv.result.preview && conv.result.result && conv.result.result.length > 0" class="preview-notice">
                  <el-alert
                    :title="`正在加载完整数据... (已显示 ${conv.result.result.length} / ${conv.result.row_count} 行)`"
                    type="info"
                    :closable="false"
                    show-icon
                  />
                </div>

                <!-- 统计信息 - 当有结果时显示 -->
                <div v-if="conv.result.result && conv.result.result.length > 0" class="result-stats">
                  <div class="stat-item">
                    <span class="stat-label">查询行数:</span>
                    <span class="stat-value">{{ conv.result.row_count || conv.result.result.length || 0 }}</span>
                  </div>
                  <div class="stat-item" v-if="conv.result.execution_time">
                    <span class="stat-label">执行时间:</span>
                    <span class="stat-value">{{ conv.result.execution_time }}秒</span>
                  </div>
                  <div class="stat-item" v-if="conv.result.preview">
                    <el-tag type="info" size="small">预览模式</el-tag>
                  </div>
                </div>

                <!-- 数据表格 - 可折叠，默认折叠 -->
                <el-collapse v-if="conv.tableData && conv.tableData.length > 0" class="result-collapse">
                  <el-collapse-item 
                    :name="`table-${index}`"
                    :title="`查询结果 (${conv.result.row_count || conv.tableData.length} 行)`"
                  >
                    <div class="result-table">
                      <el-table
                        :data="conv.tableData"
                        stripe
                        border
                        style="width: 100%"
                        max-height="400"
                        size="small"
                      >
                        <el-table-column
                          v-for="(column, colIndex) in conv.tableColumns"
                          :key="colIndex"
                          :prop="column"
                          :label="column"
                          min-width="120"
                          show-overflow-tooltip
                        />
                      </el-table>
                      
                      <div class="table-actions">
                        <el-button size="small" :icon="Download" @click="exportToExcel(conv)">导出Excel</el-button>
                        <el-button size="small" :icon="DocumentCopy" @click="copyTable(conv)">复制数据</el-button>
                      </div>
                    </div>
                  </el-collapse-item>
                </el-collapse>

                <!-- 空结果提示 -->
                <el-empty
                  v-if="!conv.loading && (!conv.tableData || conv.tableData.length === 0)"
                  description="未找到匹配的数据"
                  :image-size="80"
                />

                <!-- 图表展示区域 -->
                <div v-if="conv.result && conv.result.chart_config" class="chart-section">
                  <!-- 曲线图（默认显示） -->
                  <div v-if="conv.result.chart_config.line_chart" class="chart-container">
                    <div class="chart-header">
                      <el-icon><TrendCharts /></el-icon>
                      <span>{{ conv.result.chart_config.line_chart.title || '趋势曲线图' }}</span>
                    </div>
                    <div 
                      :id="`line-chart-${index}`" 
                      class="chart-content"
                      style="width: 100%; height: 400px;"
                    ></div>
                  </div>

                  <!-- 柱状图（可折叠，默认折叠） -->
                  <div v-if="conv.result.chart_config.bar_chart" class="chart-container chart-collapsible">
                    <el-collapse v-model="conv.chartExpanded" class="chart-collapse">
                      <el-collapse-item :name="`chart-${index}`" :title="conv.result.chart_config.bar_chart.title || '柱状图'">
                        <template #title>
                          <div class="chart-header">
                            <el-icon><Histogram /></el-icon>
                            <span>{{ conv.result.chart_config.bar_chart.title || '柱状图' }}</span>
                          </div>
                        </template>
                        <div 
                          :id="`bar-chart-${index}`" 
                          class="chart-content"
                          style="width: 100%; height: 400px;"
                        ></div>
                      </el-collapse-item>
                    </el-collapse>
                  </div>
                </div>
              </div>

              <!-- 错误信息 -->
              <div v-if="conv.error" class="error-message">
                <el-alert
                  :title="conv.error"
                  type="error"
                  :closable="false"
                  show-icon
                />
                <!-- 显示错误详情和建议 -->
                <div v-if="conv.errorDetails && conv.errorDetails.suggestion" class="error-suggestion">
                  <el-alert
                    :title="conv.errorDetails.suggestion"
                    type="info"
                    :closable="false"
                    show-icon
                    style="margin-top: 12px;"
                  />
                </div>
                <!-- 显示失败的SQL（可复制） -->
                <div v-if="conv.result && conv.result.failed_sql" class="failed-sql-box">
                  <div class="failed-sql-header">
                    <el-icon><Warning /></el-icon>
                    <span>失败的SQL（可在下一轮对话中修正）</span>
                    <el-button 
                      text 
                      size="small" 
                      @click="copySQL(conv.result.failed_sql)"
                      style="margin-left: auto;"
                    >
                      <el-icon><DocumentCopy /></el-icon>
                      复制SQL
                    </el-button>
                  </div>
                  <pre class="sql-code-display">{{ conv.result.failed_sql }}</pre>
                </div>
              </div>

              <div class="message-time">{{ formatTime(conv.responseTime) }}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- 输入区域 -->
      <div class="chat-input-area">
        <div class="input-wrapper">
          <el-input
            v-model="currentQuestion"
            type="textarea"
            :rows="2"
            placeholder="输入您的问题，例如：查询1号线昨天的客流量"
            @keyup.ctrl.enter="sendMessage(currentQuestion)"
            @keyup.enter.exact="handleEnterKey"
            class="chat-input"
            resize="none"
          />
          <div class="input-actions">
            <el-checkbox v-model="useLLM" size="small">使用AI增强</el-checkbox>
            <el-button
              type="primary"
              :icon="Search"
              :loading="loading"
              @click="sendMessage(currentQuestion)"
              :disabled="!currentQuestion.trim()"
            >
              发送
            </el-button>
          </div>
        </div>
      </div>
    </div>

    <!-- 侧边栏：历史和建议 -->
    <div class="sidebar">
      <el-card class="sidebar-card" shadow="hover">
        <template #header>
          <div class="sidebar-header">
            <el-icon><Clock /></el-icon>
            <span>查询历史</span>
          </div>
        </template>
        <div class="history-list">
          <div
            v-for="(item, index) in queryHistory"
            :key="index"
            class="history-item"
            @click="sendMessage(item)"
          >
            <el-icon><Document /></el-icon>
            <span>{{ item }}</span>
          </div>
          <el-empty v-if="queryHistory.length === 0" description="暂无历史" :image-size="60" />
        </div>
      </el-card>

      <el-card class="sidebar-card" shadow="hover" style="margin-top: 16px;">
        <template #header>
          <div class="sidebar-header">
            <el-icon><Collection /></el-icon>
            <span>数据字典</span>
          </div>
        </template>
        <el-collapse>
          <el-collapse-item title="线路列表" name="lines">
            <el-tag
              v-for="line in lines"
              :key="line"
              size="small"
              style="margin: 4px;"
              @click="sendMessage(`查询${line}的客流量`)"
            >
              {{ line }}
            </el-tag>
          </el-collapse-item>
          <el-collapse-item title="车站列表" name="stations">
            <el-scrollbar height="200px">
              <el-tag
                v-for="station in stations"
                :key="station"
                size="small"
                style="margin: 4px;"
                @click="sendMessage(`${station}今天的客流量`)"
              >
                {{ station }}
              </el-tag>
            </el-scrollbar>
          </el-collapse-item>
        </el-collapse>
      </el-card>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick, watch } from 'vue'
import { ElMessage } from 'element-plus'
import * as XLSX from 'xlsx'
import {
  Search, User, Loading, CircleCheck, CircleClose, Warning,
  Clock, Document, Collection, Download, DocumentCopy, ChatDotRound,
  Service, Operation, CopyDocument, ArrowRight, ArrowDown, Lightning,
  TrendCharts, Histogram
} from '@element-plus/icons-vue'
import * as echarts from 'echarts'
// 使用Service图标作为AI头像
const Robot = Service
import { queryAPI } from '../api/query'

const currentQuestion = ref('')
const loading = ref(false)
const useLLM = ref(true)  // 默认启用AI增强
const conversations = ref([])
const queryHistory = ref([])
const messagesContainer = ref(null)

const lines = ref([])
const stations = ref([])

const quickQueries = [
  '查询1号线昨天的客流量',
  '五一广场站今天的进站量',
  '查询最近7天各线路的客流量',
  '预测明天1号线的客流量',
  '客流量最高的10个车站'
]

// 监听图表折叠状态变化，展开时渲染柱状图
watch(() => conversations.value.map((c, idx) => ({ expanded: c.chartExpanded, index: idx })), (newVals) => {
  newVals.forEach(({ expanded, index }) => {
    const conv = conversations.value[index]
    if (conv && conv.result && conv.result.chart_config && conv.result.chart_config.bar_chart) {
      // 如果柱状图展开，渲染图表
      const chartName = `chart-${index}`
      if (Array.isArray(expanded) && expanded.includes(chartName)) {
        setTimeout(() => {
          renderCharts(index, conv)
        }, 200)
      }
    }
  })
}, { deep: true })

// 加载元数据
const loadMetadata = async () => {
  try {
    // 添加超时处理，避免卡住
    const timeoutPromise = new Promise((_, reject) => 
      setTimeout(() => reject(new Error('加载超时')), 5000)
    )
    
    const [linesRes, stationsRes] = await Promise.race([
      Promise.all([
        queryAPI.getLines().catch(() => ({ data: { lines: [] } })),
        queryAPI.getStations().catch(() => ({ data: { stations: [] } }))
      ]),
      timeoutPromise
    ])
    
    lines.value = linesRes.data?.lines || []
    stations.value = stationsRes.data?.stations || []
  } catch (error) {
    console.error('加载元数据失败:', error)
    // 即使失败也继续，使用空数组
    lines.value = []
    stations.value = []
  }
}

// 发送消息
const sendMessage = async (question) => {
  if (!question || !question.trim()) return
  
  // 防止重复提交
  if (loading.value) {
    ElMessage.warning('请等待当前查询完成')
    return
  }
  
  const questionText = question.trim()
  currentQuestion.value = ''
  
  // 获取当前时间并格式化
  const now = new Date()
  const year = now.getFullYear()
  const month = String(now.getMonth() + 1).padStart(2, '0')
  const day = String(now.getDate()).padStart(2, '0')
  const hour = String(now.getHours()).padStart(2, '0')
  const minute = String(now.getMinutes()).padStart(2, '0')
  const second = String(now.getSeconds()).padStart(2, '0')
  const currentTimeStr = `${year}年${month}月${day}日 ${hour}时${minute}分${second}秒`
  
  // 添加到对话列表（在问题前添加时间信息）
  const conversation = {
    question: `[${currentTimeStr}] ${questionText}`,
    originalQuestion: questionText,  // 保存原始问题
    timestamp: Date.now(),
    currentTime: currentTimeStr,  // 保存当前时间
    loading: true,
    response: null,
    result: null,
    processSteps: [],
    tableData: [],
    tableColumns: [],
    activeCollapse: [],
    responseTime: null,
    error: null,
    thinkingProcess: '',  // 存储思维过程
    thinkingExpanded: false,  // 默认折叠思考过程（优化：只显示结果）
    chartExpanded: [],  // 图表折叠状态（柱状图默认折叠）
    sqlExpanded: false,  // SQL预览默认折叠
    tableExpanded: false,  // 数据表格默认折叠
    scrollTimer: null  // 滚动节流定时器
  }
  conversations.value.push(conversation)
  
  // 滚动到底部
  await nextTick()
  scrollToBottom()
  
  loading.value = true
  
  // 初始化思维过程显示
  conversation.thinkingProcess = ''
  conversation.processSteps = []
  
  try {
    // 如果使用LLM，使用流式接口
    if (useLLM.value) {
      // 在问题前添加当前时间信息
      const now = new Date()
      const year = now.getFullYear()
      const month = String(now.getMonth() + 1).padStart(2, '0')
      const day = String(now.getDate()).padStart(2, '0')
      const hour = String(now.getHours()).padStart(2, '0')
      const minute = String(now.getMinutes()).padStart(2, '0')
      const second = String(now.getSeconds()).padStart(2, '0')
      const currentTimeStr = `${year}年${month}月${day}日 ${hour}时${minute}分${second}秒`
      const questionWithTime = `现在是${currentTimeStr}。${questionText}`
      
      console.log('[查询开始] 使用流式接口，问题:', questionWithTime.substring(0, 100))
      
      // 构建对话历史（包含之前的错误信息）
      const conversationHistory = []
      // 查找当前对话之前的错误信息
      for (let i = 0; i < conversations.value.length - 1; i++) {
        const prevConv = conversations.value[i]
        if (prevConv.error || (prevConv.result && prevConv.result.error)) {
          conversationHistory.push({
            question: prevConv.originalQuestion || prevConv.question,
            error: prevConv.error || prevConv.result?.error,
            failed_sql: prevConv.result?.sql || prevConv.sql,
            timestamp: prevConv.timestamp
          })
        }
      }
      
      await queryAPI.naturalLanguageQueryStream(
        questionWithTime,
        { 
          use_llm: true,
          conversation_history: conversationHistory.length > 0 ? conversationHistory : undefined
        },
        (data) => {
          // 实时处理流式数据
          console.log('[收到数据]', data.type, data)
          
          if (data.type === 'thinking_start') {
            console.log('[思考开始] 初始化思考过程')
            conversation.thinkingProcess = ''
            conversation.thinkingExpanded = false  // 默认折叠（优化：只显示结果）
            conversation.sqlExpanded = false  // SQL预览默认折叠
            conversation.tableExpanded = false  // 数据表格默认折叠
            conversation.processSteps = [{
              step: '理解问题',
              status: 'processing',
              message: '正在分析问题...',
              timestamp: Date.now() / 1000,
              expanded: false  // 默认折叠所有步骤（优化：只显示结果）
            }]
            // 强制更新显示
            nextTick(() => scrollToBottom())
          } else if (data.type === 'thinking') {
            // 实时更新思维过程（立即追加显示）
            if (data.content !== undefined && data.content !== null) {
              // 如果还没有初始化，先初始化
              if (conversation.processSteps.length === 0) {
                console.log('[思考] 首次收到思考内容，初始化步骤')
                conversation.processSteps = [{
                  step: '理解问题',
                  status: 'processing',
                  message: '正在分析问题...',
                  timestamp: Date.now() / 1000,
                  expanded: false  // 默认折叠（优化：只显示结果）
                }]
              }
              
              // 追加内容（包括空格和换行）
              const content = String(data.content)
              
              // 调试日志（记录每个消息，用于验证逐字符显示）
              const oldLength = conversation.thinkingProcess.length
              
              // 确保thinkingProcess存在（Vue响应式）
              if (!conversation.thinkingProcess) {
                conversation.thinkingProcess = ''
              }
              
              // 追加内容（使用新对象确保Vue检测到变化）
              const newThinkingProcess = (conversation.thinkingProcess || '') + content
              conversation.thinkingProcess = newThinkingProcess
              const newLength = newThinkingProcess.length
              
              // 详细日志（前100个消息详细记录，之后每50个记录一次）
              const shouldLog = newLength <= 500 || newLength % 100 === 0
              if (shouldLog) {
                const timestamp = new Date().toISOString().split('T')[1].split('.')[0]
                console.log(`[思考更新 ${timestamp}] 追加 ${content.length} 字符，总长度: ${oldLength} -> ${newLength}`, 
                  `内容: "${content.substring(0, 50)}${content.length > 50 ? '...' : ''}"`, 
                  `完整内容预览: "${newThinkingProcess.substring(0, 100)}${newThinkingProcess.length > 100 ? '...' : ''}"`)
              }
              
              // 更新步骤中的思维过程
              if (conversation.processSteps.length > 0 && conversation.processSteps[0].step === '理解问题') {
                if (!conversation.processSteps[0].details) {
                  conversation.processSteps[0].details = {}
                }
                conversation.processSteps[0].details.thinking = conversation.thinkingProcess
              }
              
              // 强制Vue立即更新DOM（不使用节流，确保每个token都立即显示）
              // 使用nextTick确保DOM更新后立即滚动
              nextTick(() => {
                try {
                  scrollToBottom()
                } catch (e) {
                  console.error('Scroll error:', e)
                }
              })
              
              // 同时使用requestAnimationFrame作为备用，确保滚动执行
              requestAnimationFrame(() => {
                try {
                  scrollToBottom()
                } catch (e) {
                  // 忽略错误
                }
              })
            } else {
              console.warn('[思考] 收到空内容:', data)
            }
          } else if (data.type === 'sql_generated') {
            // SQL生成完成（阶段1完成）
            if (conversation.processSteps.length > 0 && conversation.processSteps[0].step === '理解问题') {
              conversation.processSteps[0].status = 'success'
              conversation.processSteps[0].message = '问题分析完成'
              if (!conversation.processSteps[0].details) {
                conversation.processSteps[0].details = {}
              }
              conversation.processSteps[0].details.sql = data.sql
              if (data.thinking) {
                conversation.thinkingProcess = data.thinking
                conversation.processSteps[0].details.thinking = data.thinking
              }
            }
            
            // 添加SQL生成步骤
            conversation.processSteps.push({
              step: '生成SQL',
              status: 'success',
              message: 'SQL语句生成完成',
              details: { sql: data.sql },
              timestamp: Date.now() / 1000,
              expanded: false  // 默认折叠（优化：只显示结果）
            })
            
            // 立即显示SQL
            if (data.sql) {
              conversation.result = {
                sql: data.sql,
                result: [],
                row_count: 0
              }
            }
            
            nextTick(() => scrollToBottom())
          } else if (data.type === 'result_preview') {
            // 查询结果预览（阶段2部分完成）- 立即显示
            console.log('[阶段2预览] 收到预览数据:', data.preview_count, '/', data.total_rows)
            if (!conversation.result) {
              conversation.result = { sql: '', result: [], row_count: 0 }
            }
            conversation.result.result = data.data
            conversation.result.row_count = data.total_rows
            conversation.result.preview = true
            
            // 处理表格数据
            if (data.data && Array.isArray(data.data) && data.data.length > 0) {
              conversation.tableColumns = Object.keys(data.data[0])
              conversation.tableData = data.data
            }
            
            // 更新执行查询步骤状态（不添加新步骤，更新现有步骤）
            const execStep = conversation.processSteps.find(s => s.step === '执行查询')
            if (execStep) {
              execStep.status = 'success'
              execStep.message = `查询执行完成，返回 ${data.total_rows} 行数据（预览前 ${data.preview_count} 行）`
            }
            
            nextTick(() => scrollToBottom())
          } else if (data.type === 'result_formatted') {
            // 结果格式化完成（阶段3完成）- 立即显示
            console.log('[阶段3完成] 收到格式化数据:', data.row_count, '行')
            conversation.result = {
              sql: conversation.result?.sql || '',
              result: data.data,
              row_count: data.row_count,
              preview: false  // 取消预览模式
            }
            
            // 处理表格数据
            if (data.data && Array.isArray(data.data) && data.data.length > 0) {
              conversation.tableColumns = Object.keys(data.data[0])
              conversation.tableData = data.data
            } else {
              conversation.tableColumns = []
              conversation.tableData = []
            }
            
            // 更新或添加处理结果步骤
            let formatStep = conversation.processSteps.find(s => s.step === '处理结果')
            if (!formatStep) {
              formatStep = {
                step: '处理结果',
                status: 'processing',
                message: '正在格式化查询结果...',
                timestamp: Date.now() / 1000,
                expanded: false
              }
              conversation.processSteps.push(formatStep)
            }
            formatStep.status = 'success'
            formatStep.message = `结果格式化完成，共 ${data.row_count} 行`
            
            nextTick(() => scrollToBottom())
          } else if (data.type === 'chart_generated') {
            // 图表生成完成（阶段4完成）- 立即显示
            console.log('[阶段4完成] 收到图表配置')
            if (conversation.result) {
              conversation.result.chart_config = data.chart_config
              // 初始化图表折叠状态（柱状图默认折叠）
              conversation.chartExpanded = []
            }
            // 更新或添加图表生成步骤
            let chartStep = conversation.processSteps.find(s => s.step === '生成图表')
            if (!chartStep) {
              chartStep = {
                step: '生成图表',
                status: 'processing',
                message: '正在生成图表配置...',
                timestamp: Date.now() / 1000,
                expanded: false
              }
              conversation.processSteps.push(chartStep)
            }
            chartStep.status = 'success'
            chartStep.message = '图表配置生成完成'
            if (data.duration) {
              chartStep.duration = data.duration
            }
            // 渲染图表（延迟渲染确保DOM已更新）
            setTimeout(() => {
              renderCharts(index, conversation)
              scrollToBottom()
            }, 300)
          } else if (data.type === 'step') {
            // 更新步骤状态
            const stepIndex = conversation.processSteps.findIndex(s => s.step === data.step)
            if (stepIndex >= 0) {
              conversation.processSteps[stepIndex].status = data.status
              conversation.processSteps[stepIndex].message = data.message
            } else {
              conversation.processSteps.push({
                step: data.step,
                status: data.status,
                message: data.message,
                timestamp: Date.now() / 1000,
                expanded: false
              })
            }
            nextTick(() => scrollToBottom())
          } else if (data.type === 'complete') {
            // 查询完全完成（所有阶段都完成）
            conversation.loading = false
            conversation.response = true
            conversation.responseTime = Date.now()
            
            // 更新最终结果（可能已经部分显示了）
            conversation.result = data.data
            
            // 保存思维过程
            if (data.metadata?.thinking_process) {
              conversation.thinkingProcess = data.metadata.thinking_process
            }
            
            // 处理表格数据（如果还没有处理）
            if (data.data.result && Array.isArray(data.data.result) && data.data.result.length > 0) {
              if (!conversation.tableColumns || conversation.tableColumns.length === 0) {
                conversation.tableColumns = Object.keys(data.data.result[0])
                conversation.tableData = data.data.result
              }
            } else if (!conversation.tableData || conversation.tableData.length === 0) {
              conversation.tableColumns = []
              conversation.tableData = []
            }
            
            // 初始化图表折叠状态（如果还没有）
            if (!conversation.chartExpanded) {
              conversation.chartExpanded = []  // 柱状图默认折叠
            }
            
            // 渲染图表（如果有图表配置，延迟渲染确保DOM已更新）
            if (conversation.result && conversation.result.chart_config) {
              setTimeout(() => {
                renderCharts(index, conversation)
              }, 300)
            }
            
            // 添加到历史（使用原始问题）
            const originalQuestion = conversation.originalQuestion || questionText
            if (!queryHistory.value.includes(originalQuestion)) {
              queryHistory.value.unshift(originalQuestion)
              if (queryHistory.value.length > 20) {
                queryHistory.value.pop()
              }
            }
            
            ElMessage.success('查询成功')
            nextTick(() => scrollToBottom())
          } else if (data.type === 'error') {
            conversation.loading = false
            conversation.error = data.message
            // 保存错误详情（包含失败的SQL，供下一轮对话使用）
            if (data.details) {
              conversation.errorDetails = data.details
              // 保存失败的SQL到结果中，供下一轮对话修正使用
              if (data.sql) {
                if (!conversation.result) {
                  conversation.result = {}
                }
                conversation.result.error = data.message
                conversation.result.failed_sql = data.sql
                conversation.result.error_details = data.details
              }
            }
            // 如果错误信息包含建议，显示给用户
            if (data.details && data.details.suggestion) {
              ElMessage.warning({
                message: data.message,
                duration: 5000,
                showClose: true
              })
            } else {
              ElMessage.error(data.message)
            }
            // 即使出错，也确保思维过程已显示
            if (conversation.thinkingProcess && conversation.processSteps.length > 0) {
              conversation.processSteps[0].status = 'error'
              conversation.processSteps[0].message = data.message
              if (!conversation.processSteps[0].details) {
                conversation.processSteps[0].details = {}
              }
              conversation.processSteps[0].details.error = data.message
              if (data.details) {
                conversation.processSteps[0].details.errorDetails = data.details
              }
            }
            nextTick(() => scrollToBottom())
          } else if (data.type === 'error_detail') {
            // 保存详细的错误堆栈
            if (!conversation.errorDetails) {
              conversation.errorDetails = {}
            }
            conversation.errorDetails.traceback = data.traceback
            // 在最后一个步骤中显示错误详情
            if (conversation.processSteps.length > 0) {
              const lastStep = conversation.processSteps[conversation.processSteps.length - 1]
              if (!lastStep.details) {
                lastStep.details = {}
              }
              lastStep.details.traceback = data.traceback
            }
          }
        }
      )
    } else {
      // 非流式查询（规则引擎）
      // 在问题前添加当前时间信息
      const now = new Date()
      const year = now.getFullYear()
      const month = String(now.getMonth() + 1).padStart(2, '0')
      const day = String(now.getDate()).padStart(2, '0')
      const hour = String(now.getHours()).padStart(2, '0')
      const minute = String(now.getMinutes()).padStart(2, '0')
      const second = String(now.getSeconds()).padStart(2, '0')
      const currentTimeStr = `${year}年${month}月${day}日 ${hour}时${minute}分${second}秒`
      const questionWithTime = `现在是${currentTimeStr}。${questionText}`
      
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('请求超时，请稍后重试')), 120000)
      )
      
      // 构建对话历史（包含之前的错误信息）
      const conversationHistory = []
      for (let i = 0; i < conversations.value.length - 1; i++) {
        const prevConv = conversations.value[i]
        if (prevConv.error || (prevConv.result && prevConv.result.error)) {
          conversationHistory.push({
            question: prevConv.originalQuestion || prevConv.question,
            error: prevConv.error || prevConv.result?.error,
            failed_sql: prevConv.result?.sql || prevConv.sql,
            timestamp: prevConv.timestamp
          })
        }
      }
      
      const response = await Promise.race([
        queryAPI.naturalLanguageQuery(questionWithTime, {
          use_llm: false,
          conversation_history: conversationHistory.length > 0 ? conversationHistory : undefined
        }),
        timeoutPromise
      ])
      
      conversation.loading = false
      conversation.response = true
      conversation.responseTime = Date.now()
      
      if (response && response.code === 200 && response.data) {
        conversation.result = response.data
        
        // 保存思维过程
        conversation.thinkingProcess = response.metadata?.thinking_process || ''
        
        // 处理过程步骤
        if (response.data.process_steps && Array.isArray(response.data.process_steps) && response.data.process_steps.length > 0) {
          conversation.processSteps = response.data.process_steps.map(step => {
            const details = step.details || {}
            const thinking = details.thinking || response.metadata?.thinking_process || conversation.thinkingProcess || ''
            
            if (step.step === '理解问题' && !thinking && response.metadata?.thinking_process) {
              details.thinking = response.metadata.thinking_process
            } else if (thinking) {
              details.thinking = thinking
            }
            
            return {
              ...step,
              details: {
                ...details,
                thinking: details.thinking || thinking
              }
            }
          })
          
          if (response.metadata?.thinking_process && !conversation.thinkingProcess) {
            conversation.thinkingProcess = response.metadata.thinking_process
          }
        } else {
          conversation.processSteps = [{
            step: '理解问题',
            status: 'success',
            message: '问题分析完成',
            details: {
              thinking: response.metadata?.thinking_process || '正在分析问题...',
              intent: response.metadata?.intent,
              entities: response.metadata?.entities,
              sql: response.data?.sql
            },
            timestamp: Date.now() / 1000
          }]
          
          if (response.metadata?.thinking_process) {
            conversation.thinkingProcess = response.metadata.thinking_process
          }
        }
        
        // 处理表格数据
        if (response.data.result && Array.isArray(response.data.result) && response.data.result.length > 0) {
          conversation.tableColumns = Object.keys(response.data.result[0])
          conversation.tableData = response.data.result
        } else {
          conversation.tableColumns = []
          conversation.tableData = []
        }
        
            // 添加到历史（使用原始问题，不包含时间前缀）
            const originalQuestion = conversation.originalQuestion || questionText
            if (!queryHistory.value.includes(originalQuestion)) {
              queryHistory.value.unshift(originalQuestion)
              if (queryHistory.value.length > 20) {
                queryHistory.value.pop()
              }
            }
        
        ElMessage.success('查询成功')
      } else {
        conversation.error = response.message || '查询失败'
        ElMessage.error(conversation.error)
      }
    }
  } catch (error) {
    conversation.loading = false
    conversation.response = true
    conversation.responseTime = Date.now()
    // 清理滚动定时器
    if (conversation.scrollTimer) {
      clearTimeout(conversation.scrollTimer)
      conversation.scrollTimer = null
    }
    conversation.error = error.message || '查询失败，请检查网络连接'
    
    // 确保processSteps是数组
    if (!Array.isArray(conversation.processSteps)) {
      conversation.processSteps = []
    }
    
    conversation.processSteps.push({
      step: '查询失败',
      status: 'error',
      message: conversation.error,
      timestamp: Date.now() / 1000,
      expanded: false  // 默认折叠（优化：只显示结果，错误时用户可手动展开查看）
    })
    
    ElMessage.error(conversation.error)
    console.error('查询错误:', error)
  } finally {
    loading.value = false
    try {
      await nextTick()
      scrollToBottom()
    } catch (e) {
      console.warn('滚动失败:', e)
    }
  }
}

// 滚动到底部
const scrollToBottom = () => {
  if (messagesContainer.value) {
    try {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    } catch (error) {
      console.warn('滚动失败:', error)
    }
  }
}

// 格式化时间
const formatTime = (timestamp) => {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  return date.toLocaleTimeString('zh-CN', { hour12: false })
}

// 检查是否有详情
const hasDetails = (details) => {
  if (!details) return false
  return !!(details.intent || details.entities || details.thinking || details.sql)
}

// 检查是否有思维过程
const hasThinking = (step, conv) => {
  return !!(step.details?.thinking || (step.step === '理解问题' && conv.thinkingProcess))
}

// 切换思考节点展开/折叠（Dify风格）
const toggleThinkingNode = (convIndex) => {
  const conv = conversations.value[convIndex]
  if (conv) {
    conv.thinkingExpanded = !conv.thinkingExpanded
  }
}

// 切换步骤节点展开/折叠（Dify风格）
const toggleStepNode = (convIndex, stepIndex) => {
  const conv = conversations.value[convIndex]
  if (conv && conv.processSteps && conv.processSteps[stepIndex]) {
    const step = conv.processSteps[stepIndex]
    step.expanded = step.expanded === undefined ? false : !step.expanded
  }
}

// 格式化思考过程为行（Dify风格）
const formatThinkingLines = (thinkingText) => {
  if (!thinkingText) return []
  return thinkingText.split('\n').filter(line => line.trim())
}

// 格式化思考文本（支持逐字符显示，保留换行和格式）
const formatThinkingText = (thinkingText) => {
  if (!thinkingText) return ''
  // 转义HTML特殊字符，保留换行
  return thinkingText
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>')
}

// 格式化步骤时间
const formatStepTime = (timestamp) => {
  if (!timestamp) return ''
  const date = new Date(timestamp * 1000)
  const hours = String(date.getHours()).padStart(2, '0')
  const minutes = String(date.getMinutes()).padStart(2, '0')
  const seconds = String(date.getSeconds()).padStart(2, '0')
  return `${hours}:${minutes}:${seconds}`
}

// 渲染图表
const renderCharts = (convIndex, conversation) => {
  if (!conversation.result || !conversation.result.chart_config) {
    return
  }
  
  const chartConfig = conversation.result.chart_config
  
  // 渲染曲线图（默认显示）
  if (chartConfig.line_chart && chartConfig.line_chart.config) {
    setTimeout(() => {
      const lineChartId = `line-chart-${convIndex}`
      const lineChartEl = document.getElementById(lineChartId)
      if (lineChartEl) {
        // 如果已经初始化过，先销毁
        const existingChart = echarts.getInstanceByDom(lineChartEl)
        if (existingChart) {
          existingChart.dispose()
        }
        
        const lineChart = echarts.init(lineChartEl)
        lineChart.setOption(chartConfig.line_chart.config)
        
        // 响应式调整
        const resizeHandler = () => {
          lineChart.resize()
        }
        window.addEventListener('resize', resizeHandler)
        
        // 保存resize handler以便后续清理
        if (!conversation._chartResizeHandlers) {
          conversation._chartResizeHandlers = []
        }
        conversation._chartResizeHandlers.push({ chart: lineChart, handler: resizeHandler })
      }
    }, 100)  // 延迟确保DOM已渲染
  }
  
  // 渲染柱状图（可折叠，展开时渲染）
  if (chartConfig.bar_chart && chartConfig.bar_chart.config) {
    setTimeout(() => {
      const barChartId = `bar-chart-${convIndex}`
      const barChartEl = document.getElementById(barChartId)
      if (barChartEl) {
        // 如果已经初始化过，先销毁
        const existingChart = echarts.getInstanceByDom(barChartEl)
        if (existingChart) {
          existingChart.dispose()
        }
        
        const barChart = echarts.init(barChartEl)
        barChart.setOption(chartConfig.bar_chart.config)
        
        // 响应式调整
        const resizeHandler = () => {
          barChart.resize()
        }
        window.addEventListener('resize', resizeHandler)
        
        // 保存resize handler以便后续清理
        if (!conversation._chartResizeHandlers) {
          conversation._chartResizeHandlers = []
        }
        conversation._chartResizeHandlers.push({ chart: barChart, handler: resizeHandler })
      }
    }, 200)  // 延迟稍长，确保折叠面板已渲染
  }
}

// 处理Enter键
const handleEnterKey = (e) => {
  if (e.ctrlKey || e.shiftKey) return
  e.preventDefault()
  if (currentQuestion.value.trim() && !loading.value) {
    sendMessage(currentQuestion.value)
  }
}

// 处理滚动
const handleScroll = () => {
  // 可以在这里添加滚动相关的逻辑
}

// 导出Excel
const exportToExcel = (conv) => {
  try {
    if (!conv.tableData || conv.tableData.length === 0) {
      ElMessage.warning('没有数据可导出')
      return
    }

    // 准备数据：将对象数组转换为二维数组
    const headers = conv.tableColumns || []
    const data = [headers] // 第一行是表头
    
    // 添加数据行
    conv.tableData.forEach(row => {
      const rowData = headers.map(col => {
        const value = row[col]
        // 处理 null/undefined
        if (value === null || value === undefined) {
          return ''
        }
        // 处理日期类型
        if (value instanceof Date) {
          return value.toLocaleString('zh-CN')
        }
        // 处理数字，保留精度
        if (typeof value === 'number') {
          return value
        }
        // 其他类型转为字符串
        return String(value)
      })
      data.push(rowData)
    })

    // 创建工作簿
    const wb = XLSX.utils.book_new()
    
    // 创建工作表（使用数组转工作表的方法）
    const ws = XLSX.utils.aoa_to_sheet(data)
    
    // 设置列宽（自动调整）
    const colWidths = headers.map((col) => {
      // 计算该列的最大宽度
      let maxLength = col.length
      conv.tableData.forEach(row => {
        const cellValue = String(row[col] || '')
        if (cellValue.length > maxLength) {
          maxLength = cellValue.length
        }
      })
      // 设置列宽，最小10，最大50
      return { wch: Math.min(Math.max(maxLength + 2, 10), 50) }
    })
    ws['!cols'] = colWidths
    
    // 将工作表添加到工作簿
    const sheetName = '查询结果'
    XLSX.utils.book_append_sheet(wb, ws, sheetName)
    
    // 生成文件名（包含时间戳）
    const now = new Date()
    const timestamp = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}`
    const question = conv.originalQuestion || conv.question || '查询结果'
    // 清理文件名中的非法字符
    const safeQuestion = question.replace(/[<>:"/\\|?*]/g, '').substring(0, 30)
    const fileName = `${safeQuestion}_${timestamp}.xlsx`
    
    // 导出文件
    XLSX.writeFile(wb, fileName)
    
    ElMessage.success(`Excel文件已导出：${fileName}`)
  } catch (error) {
    console.error('导出Excel失败:', error)
    ElMessage.error(`导出失败: ${error.message || '未知错误'}`)
  }
}

// 复制表格
const copyTable = async (conv) => {
  try {
    const text = conv.tableData.map(row => 
      conv.tableColumns.map(col => row[col]).join('\t')
    ).join('\n')
    
    await navigator.clipboard.writeText(text)
    ElMessage.success('数据已复制到剪贴板')
  } catch (error) {
    ElMessage.error('复制失败')
  }
}

// 复制SQL
const copySQL = async (sql) => {
  try {
    await navigator.clipboard.writeText(sql)
    ElMessage.success('SQL已复制到剪贴板')
  } catch (error) {
    ElMessage.error('复制失败')
  }
}

onMounted(() => {
  // 延迟加载元数据，避免阻塞页面渲染
  setTimeout(() => {
    loadMetadata()
  }, 100)
  
  // 监听滚动，自动滚动到底部
  if (messagesContainer.value) {
    const observer = new MutationObserver(() => {
      scrollToBottom()
    })
    observer.observe(messagesContainer.value, {
      childList: true,
      subtree: true
    })
  }
})
</script>

<style scoped>
.chat-page {
  display: flex;
  height: calc(100vh - 120px);
  gap: 16px;
  max-width: 1600px;
  margin: 0 auto;
}

.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #ffffff;
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1), 0 4px 16px rgba(0, 0, 0, 0.08);
  overflow: hidden;
  border: 1px solid #e5e7eb;
  position: relative;
}

.chat-container::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, #409eff 0%, #66b1ff 50%, #85d0ff 100%);
  z-index: 10;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 32px;
  background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
  position: relative;
}

.chat-messages::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, rgba(0, 0, 0, 0.05) 50%, transparent 100%);
}

/* 欢迎消息 */
.welcome-message {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  min-height: 400px;
}

.welcome-content {
  text-align: center;
  max-width: 600px;
  animation: fadeInScale 0.6s ease-out;
}

@keyframes fadeInScale {
  from {
    opacity: 0;
    transform: scale(0.95);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

.welcome-icon {
  color: #409eff;
  margin-bottom: 20px;
  filter: drop-shadow(0 4px 12px rgba(64, 158, 255, 0.3));
  animation: float 3s ease-in-out infinite;
}

@keyframes float {
  0%, 100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-10px);
  }
}

.welcome-content h2 {
  color: #1f2937;
  margin-bottom: 16px;
  font-size: 28px;
  font-weight: 800;
  background: linear-gradient(135deg, #1f2937 0%, #409eff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: -0.5px;
}

.welcome-content p {
  color: #64748b;
  margin-bottom: 32px;
  font-size: 16px;
  font-weight: 500;
  line-height: 1.7;
}

.quick-suggestions {
  margin-top: 32px;
}

.suggestion-title {
  color: #6b7280;
  font-size: 14px;
  margin-bottom: 12px;
  font-weight: 600;
}

.suggestion-tag {
  margin: 6px;
  padding: 10px 18px;
  cursor: pointer;
  transition: all 0.3s;
  border: 2px solid #d1d5db;
  background: #ffffff;
  color: #374151;
  font-weight: 500;
  font-size: 13px;
}

.suggestion-tag:hover {
  background: #409eff;
  color: #ffffff;
  border-color: #409eff;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(64, 158, 255, 0.3);
}

/* 对话消息 */
.conversation-item {
  margin-bottom: 32px;
  animation: fadeInUp 0.4s ease-out;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.message {
  display: flex;
  margin-bottom: 20px;
  animation: slideIn 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateX(-10px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

.user-message {
  justify-content: flex-end;
}

.ai-message {
  justify-content: flex-start;
}

.message-avatar {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  font-size: 22px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  z-index: 1;
}

.message-avatar::before {
  content: '';
  position: absolute;
  inset: -2px;
  border-radius: 50%;
  padding: 2px;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.1));
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  opacity: 0;
  transition: opacity 0.3s;
}

.message:hover .message-avatar::before {
  opacity: 1;
}

.user-avatar {
  background: linear-gradient(135deg, #409eff 0%, #66b1ff 50%, #85d0ff 100%);
  color: #ffffff;
  margin-left: 12px;
  box-shadow: 0 4px 16px rgba(64, 158, 255, 0.35), 0 2px 8px rgba(64, 158, 255, 0.2);
}

.user-avatar:hover {
  transform: scale(1.05);
  box-shadow: 0 6px 20px rgba(64, 158, 255, 0.45), 0 4px 12px rgba(64, 158, 255, 0.3);
}

.ai-avatar {
  background: linear-gradient(135deg, #67c23a 0%, #85ce61 50%, #95d475 100%);
  color: #ffffff;
  margin-right: 12px;
  box-shadow: 0 4px 16px rgba(103, 194, 58, 0.35), 0 2px 8px rgba(103, 194, 58, 0.2);
}

.ai-avatar:hover {
  transform: scale(1.05);
  box-shadow: 0 6px 20px rgba(103, 194, 58, 0.45), 0 4px 12px rgba(103, 194, 58, 0.3);
}

.message-content {
  max-width: 75%;
  min-width: 200px;
}

.user-content {
  background: linear-gradient(135deg, #409eff 0%, #66b1ff 50%, #85d0ff 100%);
  color: #ffffff;
  padding: 14px 18px;
  border-radius: 20px 20px 6px 20px;
  box-shadow: 0 4px 16px rgba(64, 158, 255, 0.3), 0 2px 8px rgba(64, 158, 255, 0.2);
  position: relative;
  overflow: hidden;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.user-content::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, transparent 100%);
  pointer-events: none;
}

.user-content:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(64, 158, 255, 0.4), 0 4px 12px rgba(64, 158, 255, 0.3);
}

.time-prefix {
  color: rgba(255, 255, 255, 0.95);
  font-size: 12px;
  font-weight: 600;
  margin-right: 8px;
  opacity: 0.9;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

.ai-content {
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
  color: #1f2937;
  padding: 20px;
  border-radius: 20px 20px 20px 6px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08), 0 2px 8px rgba(0, 0, 0, 0.04);
  border: 1px solid #e5e7eb;
  position: relative;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.ai-content::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: linear-gradient(90deg, #67c23a 0%, #85ce61 50%, #95d475 100%);
  border-radius: 20px 20px 0 0;
  opacity: 0.6;
}

.ai-content:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.12), 0 4px 12px rgba(0, 0, 0, 0.08);
  border-color: #d1d5db;
}

.message-text {
  line-height: 1.7;
  word-wrap: break-word;
  font-size: 14.5px;
  font-weight: 400;
  letter-spacing: 0.2px;
  position: relative;
  z-index: 1;
}

.user-content .message-text {
  color: #ffffff;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

.message-time {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.85);
  margin-top: 8px;
  text-align: right;
  font-weight: 500;
  opacity: 0.9;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

.ai-content .message-time {
  color: #9ca3af;
  text-align: left;
  margin-top: 10px;
  font-weight: 500;
}

/* 加载指示器 */
.loading-indicator {
  display: flex;
  align-items: center;
  gap: 10px;
  color: #6b7280;
  padding: 12px 16px;
  background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
  border-radius: 10px;
  border: 1px solid #bfdbfe;
  font-weight: 500;
  font-size: 14px;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.7;
  }
}

/* 思维链容器（Dify风格 - 美化版） */
.thinking-chain-container {
  margin: 16px 0;
}

/* 思考节点（Dify风格 - 树状结构，美化版） */
.thinking-node {
  margin-bottom: 12px;
  border-radius: 12px;
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
  border: 1px solid #e5e7eb;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.05);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  overflow: hidden;
}

.thinking-node:hover {
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15), 0 2px 6px rgba(0, 0, 0, 0.1);
  transform: translateY(-1px);
}

.thinking-node-main {
  border-left: 4px solid #3b82f6;
  background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 50%, #f8fafc 100%);
  position: relative;
}

.thinking-node-main::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 50%, #93c5fd 100%);
}

.thinking-node-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 18px;
  cursor: pointer;
  user-select: none;
  transition: all 0.2s ease;
  background: rgba(255, 255, 255, 0.6);
  backdrop-filter: blur(10px);
}

.thinking-node-header:hover {
  background: rgba(255, 255, 255, 0.9);
  padding-left: 20px;
}

.thinking-node-left {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
}

.thinking-node-icon {
  font-size: 14px;
  color: #6b7280;
  transition: transform 0.2s ease;
}

.thinking-node-icon.is-expanded {
  transform: rotate(0deg);
}

.thinking-node-status {
  font-size: 18px;
  color: #3b82f6;
  filter: drop-shadow(0 2px 4px rgba(59, 130, 246, 0.3));
}

.thinking-loading-icon {
  animation: rotate 1s linear infinite;
  filter: drop-shadow(0 2px 4px rgba(59, 130, 246, 0.3));
}

.thinking-node-title {
  font-weight: 700;
  font-size: 15px;
  color: #1e40af;
  letter-spacing: 0.3px;
}

.thinking-node-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.thinking-node-count {
  font-size: 12px;
  color: #6b7280;
  font-weight: normal;
}

.thinking-node-content {
  padding: 0 18px 18px 18px;
  border-top: 1px solid rgba(59, 130, 246, 0.1);
  margin-top: 0;
  background: rgba(255, 255, 255, 0.5);
  animation: slideDown 0.3s ease-out;
}

@keyframes slideDown {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.thinking-content-text {
  padding-top: 16px;
  font-size: 13.5px;
  line-height: 1.9;
  color: #1f2937;
  font-weight: 400;
}

.thinking-text-display {
  color: #374151;
  word-break: break-word;
  white-space: pre-wrap;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  line-height: 1.9;
  /* 确保逐字符显示流畅 */
  will-change: contents;
  /* 优化渲染性能 */
  contain: layout style;
  /* 确保即使内容很少也能显示 */
  min-height: 1em;
}

/* 当有内容时，隐藏占位符 */
.thinking-content-text:has(.thinking-text-display:not(:empty)) .thinking-placeholder {
  display: none;
}

.thinking-text-line {
  display: flex;
  gap: 12px;
  margin-bottom: 4px;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
}

.thinking-line-number {
  color: #9ca3af;
  font-size: 12px;
  min-width: 24px;
  text-align: right;
  user-select: none;
}

.thinking-line-content {
  flex: 1;
  white-space: pre-wrap;
  word-break: break-word;
  color: #1f2937;
}

.thinking-placeholder {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #6b7280;
  font-style: italic;
  padding: 20px 0;
}

.thinking-placeholder-icon {
  animation: rotate 1s linear infinite;
}

@keyframes rotate {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

/* 过程步骤树（Dify风格 - 美化版） */
.process-steps-tree {
  margin: 20px 0;
  position: relative;
  padding-left: 8px;
}

.process-steps-tree::before {
  content: '';
  position: absolute;
  left: 24px;
  top: 0;
  bottom: 0;
  width: 3px;
  background: linear-gradient(180deg, #3b82f6 0%, #60a5fa 30%, #93c5fd 60%, transparent 100%);
  z-index: 0;
  border-radius: 2px;
  box-shadow: 0 0 8px rgba(59, 130, 246, 0.2);
}

.process-step-node {
  margin-bottom: 16px;
  border-radius: 14px;
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
  border: 1px solid #e5e7eb;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08), 0 2px 8px rgba(0, 0, 0, 0.04);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border-left: 5px solid #d1d5db;
  position: relative;
  z-index: 1;
  overflow: hidden;
  margin-left: 8px;
}

.process-step-node::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: transparent;
  transition: all 0.3s ease;
  border-radius: 14px 14px 0 0;
}

.process-step-node:hover {
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12), 0 4px 12px rgba(0, 0, 0, 0.08);
  transform: translateY(-3px) scale(1.01);
  border-color: #cbd5e1;
}

.process-step-node.step-processing {
  border-left-color: #3b82f6;
  background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 50%, #f0f9ff 100%);
  animation: pulseGlow 2s ease-in-out infinite;
}

.process-step-node.step-processing::before {
  background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 50%, #93c5fd 100%);
}

@keyframes pulseGlow {
  0%, 100% {
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.2), 0 1px 3px rgba(0, 0, 0, 0.05);
  }
  50% {
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4), 0 2px 6px rgba(59, 130, 246, 0.2);
  }
}

.process-step-node.step-success {
  border-left-color: #10b981;
  background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 50%, #f0fdf4 100%);
}

.process-step-node.step-success::before {
  background: linear-gradient(90deg, #10b981 0%, #34d399 50%, #6ee7b7 100%);
}

.process-step-node.step-error {
  border-left-color: #ef4444;
  background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 50%, #fef2f2 100%);
}

.process-step-node.step-error::before {
  background: linear-gradient(90deg, #ef4444 0%, #f87171 50%, #fca5a5 100%);
}

.process-step-node.step-warning {
  border-left-color: #f59e0b;
  background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 50%, #fffbeb 100%);
}

.process-step-node.step-warning::before {
  background: linear-gradient(90deg, #f59e0b 0%, #fbbf24 50%, #fcd34d 100%);
}

.step-node-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 18px;
  cursor: pointer;
  user-select: none;
  transition: all 0.2s ease;
  background: rgba(255, 255, 255, 0.6);
  backdrop-filter: blur(10px);
}

.step-node-header:hover {
  background: rgba(255, 255, 255, 0.9);
  padding-left: 20px;
}

.step-node-left {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
}

.step-expand-icon {
  font-size: 14px;
  color: #6b7280;
  transition: transform 0.2s ease;
}

.step-expand-icon.is-expanded {
  transform: rotate(0deg);
}

.step-status-icon {
  font-size: 18px;
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.15));
}

.step-status-icon.status-processing {
  color: #3b82f6;
  animation: pulse 2s ease-in-out infinite;
  filter: drop-shadow(0 2px 6px rgba(59, 130, 246, 0.4));
}

.step-status-icon.status-success {
  color: #10b981;
  filter: drop-shadow(0 2px 6px rgba(16, 185, 129, 0.3));
}

.step-status-icon.status-error {
  color: #ef4444;
  filter: drop-shadow(0 2px 6px rgba(239, 68, 68, 0.3));
}

.step-status-icon.status-warning {
  color: #f59e0b;
  filter: drop-shadow(0 2px 6px rgba(245, 158, 11, 0.3));
}

.step-node-title {
  font-weight: 700;
  font-size: 15px;
  color: #1f2937;
  letter-spacing: 0.3px;
}

.step-node-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.step-timestamp {
  font-size: 11px;
  color: #9ca3af;
  font-weight: normal;
}

.step-node-content {
  padding: 0 20px 20px 20px;
  border-top: 1px solid rgba(0, 0, 0, 0.08);
  margin-top: 0;
  background: rgba(255, 255, 255, 0.6);
  animation: slideDown 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
}

.step-node-content::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, rgba(0, 0, 0, 0.05) 50%, transparent 100%);
}

.step-message {
  padding-top: 18px;
  font-size: 14px;
  color: #1f2937;
  line-height: 1.9;
  font-weight: 500;
  letter-spacing: 0.2px;
}

.step-processing {
  border-left-color: #409eff;
  background: #e6f2ff;
  color: #1f2937;
}

.step-success {
  border-left-color: #67c23a;
  background: #f0f9ff;
  color: #1f2937;
}

.step-error {
  border-left-color: #f56c6c;
  background: #fef0f0;
  color: #1f2937;
}

.step-warning {
  border-left-color: #e6a23c;
  background: #fdf6ec;
  color: #1f2937;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

/* 图表区域 */
.chart-section {
  margin: 20px 0;
}

.chart-container {
  margin-bottom: 20px;
  padding: 16px;
  background: #ffffff;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.chart-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
  font-weight: 600;
  font-size: 16px;
  color: #1f2937;
}

.chart-header .el-icon {
  font-size: 20px;
  color: #3b82f6;
}

.chart-content {
  min-height: 400px;
}

.chart-collapsible {
  margin-top: 16px;
}

.chart-collapse {
  border: none;
}

.chart-collapse .el-collapse-item__header {
  padding: 12px 16px;
  background: #f8fafc;
  border-radius: 6px;
  font-weight: 600;
}

.chart-collapse .el-collapse-item__content {
  padding: 16px 0;
}

/* 错误建议 */
.error-suggestion {
  margin-top: 12px;
}

/* 失败的SQL显示 */
.failed-sql-box {
  margin-top: 16px;
  padding: 20px;
  background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
  border-radius: 12px;
  border: 2px solid #fecaca;
  box-shadow: 0 4px 16px rgba(239, 68, 68, 0.1);
  position: relative;
  overflow: hidden;
}

.failed-sql-box::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: linear-gradient(90deg, #ef4444 0%, #f87171 50%, #fca5a5 100%);
}

.failed-sql-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 16px;
  font-weight: 700;
  color: #991b1b;
  font-size: 15px;
  padding-bottom: 12px;
  border-bottom: 2px solid #fecaca;
}

.failed-sql-header .el-icon {
  font-size: 18px;
}

.step-details-collapse {
  margin-top: 8px;
}

.thinking-box {
  margin: 16px 0;
  padding: 18px 20px;
  background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
  border-radius: 12px;
  border-left: 5px solid #3b82f6;
  border: 1px solid #bfdbfe;
  box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
  position: relative;
  overflow: hidden;
}

.thinking-box::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 50%, #93c5fd 100%);
}

.thinking-title {
  font-weight: 700;
  color: #1e40af;
  margin-bottom: 12px;
  font-size: 15px;
  letter-spacing: 0.3px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.thinking-content {
  font-size: 13.5px;
  color: #1f2937;
  line-height: 1.9;
  white-space: pre-wrap;
  word-break: break-word;
  font-weight: 400;
  letter-spacing: 0.2px;
}

.detail-item {
  margin: 12px 0;
  font-size: 13px;
}

.detail-item strong {
  color: #1f2937;
  margin-right: 8px;
  font-weight: 600;
}

.detail-item pre {
  margin-top: 10px;
  padding: 16px;
  background: linear-gradient(135deg, #282c34 0%, #1e222a 100%);
  color: #abb2bf;
  border-radius: 10px;
  overflow-x: auto;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 12.5px;
  line-height: 1.8;
  border: 1px solid #3a3f4b;
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.3);
  position: relative;
}

.detail-item pre::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 4px;
  height: 100%;
  background: linear-gradient(180deg, #3b82f6 0%, #60a5fa 100%);
  border-radius: 10px 0 0 10px;
}

.sql-code {
  margin: 0;
  padding: 12px;
  background: #282c34;
  color: #abb2bf;
  border-radius: 6px;
  overflow-x: auto;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 12px;
  line-height: 1.6;
}

/* SQL预览卡片 */
.sql-preview-card {
  margin: 20px 0;
  padding: 20px;
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
  border-radius: 14px;
  border: 2px solid #334155;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3), 0 2px 8px rgba(0, 0, 0, 0.2);
  position: relative;
  overflow: hidden;
}

.sql-preview-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 50%, #93c5fd 100%);
}

.sql-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 16px;
  padding-bottom: 14px;
  border-bottom: 2px solid #334155;
  color: #e2e8f0;
  font-weight: 700;
  font-size: 16px;
  letter-spacing: 0.3px;
}

.sql-header .el-icon {
  color: #60a5fa;
  font-size: 18px;
}

.sql-code-display {
  margin: 0;
  padding: 18px 20px;
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  color: #cbd5e1;
  border-radius: 10px;
  overflow-x: auto;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 13px;
  line-height: 1.9;
  border: 1px solid #334155;
  white-space: pre-wrap;
  word-break: break-word;
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.3);
  position: relative;
}

.sql-code-display::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 4px;
  height: 100%;
  background: linear-gradient(180deg, #3b82f6 0%, #60a5fa 100%);
  border-radius: 10px 0 0 10px;
}

/* 查询结果 */
.query-result {
  margin-top: 12px;
}

.result-collapse {
  margin: 12px 0;
  margin-bottom: 12px;
  
  .el-collapse-item__header {
    padding: 12px 16px;
    font-weight: 500;
    background-color: var(--el-bg-color-page);
    border-radius: 4px;
  }
  
  .el-collapse-item__content {
    padding: 16px;
  }
}

.sql-header-collapse {
  display: flex;
  align-items: center;
  width: 100%;
  
  .el-icon {
    margin-right: 8px;
  }
}

.result-stats {
  display: flex;
  gap: 40px;
  padding: 20px 24px;
  background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
  border-radius: 12px;
  margin-bottom: 16px;
  border: 1px solid #bfdbfe;
  box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
  position: relative;
  overflow: hidden;
}

.result-stats::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 50%, #93c5fd 100%);
}

.stat-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.stat-item {
  position: relative;
  padding-left: 12px;
}

.stat-item::before {
  content: '';
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  width: 3px;
  height: 20px;
  background: linear-gradient(180deg, #3b82f6 0%, #60a5fa 100%);
  border-radius: 2px;
}

.stat-label {
  color: #64748b;
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 4px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.stat-value {
  color: #1e40af;
  font-weight: 800;
  font-size: 20px;
  line-height: 1.2;
  text-shadow: 0 1px 2px rgba(30, 64, 175, 0.1);
}

.result-table {
  margin-top: 16px;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
  border: 1px solid #e5e7eb;
  background: #ffffff;
}

.result-table :deep(.el-table) {
  border-radius: 12px;
}

.result-table :deep(.el-table__header) {
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
}

.result-table :deep(.el-table th) {
  background: transparent !important;
  color: #1f2937;
  font-weight: 700;
  font-size: 13px;
  border-bottom: 2px solid #e5e7eb;
}

.result-table :deep(.el-table td) {
  border-bottom: 1px solid #f1f5f9;
}

.result-table :deep(.el-table tr:hover > td) {
  background: #f8fafc !important;
}

.table-actions {
  margin-top: 16px;
  padding: 12px 16px;
  background: #f8fafc;
  border-top: 1px solid #e5e7eb;
  display: flex;
  gap: 10px;
  border-radius: 0 0 12px 12px;
}

.table-actions .el-button {
  border-radius: 8px;
  font-weight: 500;
  transition: all 0.3s;
}

.table-actions .el-button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.error-message {
  margin-top: 16px;
  animation: shake 0.5s ease-in-out;
}

@keyframes shake {
  0%, 100% { transform: translateX(0); }
  10%, 30%, 50%, 70%, 90% { transform: translateX(-4px); }
  20%, 40%, 60%, 80% { transform: translateX(4px); }
}

.error-message :deep(.el-alert) {
  border-radius: 12px;
  box-shadow: 0 4px 16px rgba(239, 68, 68, 0.15);
  border: 1px solid #fecaca;
  overflow: hidden;
}

.error-message :deep(.el-alert__title) {
  font-weight: 600;
  font-size: 14px;
}

/* 输入区域 */
.chat-input-area {
  border-top: 2px solid #f1f5f9;
  padding: 20px 24px;
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
  box-shadow: 0 -4px 16px rgba(0, 0, 0, 0.05);
  position: relative;
}

.chat-input-area::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, rgba(0, 0, 0, 0.05) 50%, transparent 100%);
}

.input-wrapper {
  max-width: 100%;
}

.chat-input {
  margin-bottom: 12px;
}

.chat-input :deep(.el-textarea__inner) {
  border-radius: 14px;
  border: 2px solid #e5e7eb;
  padding: 14px 18px;
  font-size: 14.5px;
  line-height: 1.7;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  background: #ffffff;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.chat-input :deep(.el-textarea__inner):focus {
  border-color: #409eff;
  box-shadow: 0 0 0 4px rgba(64, 158, 255, 0.12), 0 4px 16px rgba(64, 158, 255, 0.15);
  background: #ffffff;
  transform: translateY(-1px);
}

.input-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

/* 侧边栏 */
.sidebar {
  width: 280px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.sidebar-card {
  border-radius: 12px;
}

.sidebar-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 700;
  color: #1f2937;
  font-size: 15px;
}

.history-list {
  max-height: 400px;
  overflow-y: auto;
}

.history-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 14px;
  margin-bottom: 6px;
  background: #f9fafb;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s;
  font-size: 13px;
  color: #374151;
  font-weight: 500;
  border: 1px solid #e5e7eb;
}

.history-item:hover {
  background: #eff6ff;
  color: #1d4ed8;
  border-color: #93c5fd;
  transform: translateX(4px);
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.15);
}

.history-item .el-icon {
  color: #6b7280;
}

.history-item:hover .el-icon {
  color: #1d4ed8;
}

/* 响应式 */
@media (max-width: 1200px) {
  .sidebar {
    display: none;
  }
}

@media (max-width: 768px) {
  .chat-page {
    height: calc(100vh - 100px);
  }
  
  .message-content {
    max-width: 85%;
  }
  
  .welcome-content h2 {
    font-size: 20px;
  }
}
</style>
