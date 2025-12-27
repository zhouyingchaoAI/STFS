<template>
  <el-container class="app-container">
    <el-header class="app-header">
      <div class="header-content">
        <div class="logo">
          <el-icon :size="28"><Location /></el-icon>
          <span class="title">地铁客流智能问数系统</span>
        </div>
        <div class="header-actions">
          <el-button text @click="toggleTheme">
            <el-icon><Sunny v-if="isDark" /><Moon v-else /></el-icon>
          </el-button>
        </div>
      </div>
    </el-header>
    
    <el-main class="app-main">
      <router-view />
    </el-main>
    
    <el-footer class="app-footer">
      <div class="footer-content">
        <span>© 2024 地铁客流智能问数系统 | Powered by FastAPI + Vue 3</span>
      </div>
    </el-footer>
  </el-container>
</template>

<script setup>
import { ref, onMounted } from 'vue'

console.log('App.vue script setup loaded')

const isDark = ref(false)

const toggleTheme = () => {
  isDark.value = !isDark.value
  document.documentElement.classList.toggle('dark', isDark.value)
}

onMounted(() => {
  console.log('App.vue mounted')
  // 检查系统主题偏好
  if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
    isDark.value = true
    document.documentElement.classList.add('dark')
  }
})
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

#app {
  width: 100%;
  height: 100vh;
}

.app-container {
  height: 100vh;
  background: var(--el-bg-color-page, #f5f7fa);
}

.app-header {
  background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e8ba3 100%);
  color: white;
  padding: 0;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
  position: relative;
  overflow: hidden;
}

.app-header::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: url('data:image/svg+xml,<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg"><defs><pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
  opacity: 0.3;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 100%;
  padding: 0 32px;
  position: relative;
  z-index: 1;
}

.logo {
  display: flex;
  align-items: center;
  gap: 16px;
  font-size: 22px;
  font-weight: 700;
  letter-spacing: 0.5px;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.logo .el-icon {
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.2));
}

.title {
  font-size: 22px;
  font-weight: 700;
  letter-spacing: 0.5px;
}

.app-main {
  padding: 24px;
  background: #f3f4f6;
  overflow-y: auto;
  min-height: calc(100vh - 120px);
}

.app-footer {
  background: var(--el-bg-color, #ffffff);
  border-top: 1px solid var(--el-border-color, #dcdfe6);
  padding: 16px 0;
  text-align: center;
  color: var(--el-text-color-secondary, #909399);
  font-size: 14px;
}

.footer-content {
  padding: 0 24px;
}
</style>

