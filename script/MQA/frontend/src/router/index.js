import { createRouter, createWebHistory } from 'vue-router'
import QueryPage from '../views/QueryPage.vue'

const routes = [
  {
    path: '/',
    name: 'Query',
    component: QueryPage
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router

