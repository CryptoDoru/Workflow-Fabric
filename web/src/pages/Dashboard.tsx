import { useQuery } from '@tanstack/react-query'
import { Bot, GitBranch, Play, AlertTriangle, CheckCircle, XCircle } from 'lucide-react'
import { Link } from 'react-router-dom'

interface Stats {
  agents: number
  workflows: number
  executions: number
  successRate: number
}

interface RecentExecution {
  id: string
  workflow_id: string
  workflow_name: string
  status: 'completed' | 'failed' | 'running'
  duration_ms: number
  started_at: string
}

async function fetchStats(): Promise<Stats> {
  // In production, this would call the API
  return {
    agents: 12,
    workflows: 5,
    executions: 847,
    successRate: 94.2,
  }
}

async function fetchRecentExecutions(): Promise<RecentExecution[]> {
  return [
    {
      id: 'exec-001',
      workflow_id: 'research-pipeline',
      workflow_name: 'Research Pipeline',
      status: 'completed',
      duration_ms: 2340,
      started_at: new Date(Date.now() - 5 * 60000).toISOString(),
    },
    {
      id: 'exec-002',
      workflow_id: 'code-review',
      workflow_name: 'Code Review Pipeline',
      status: 'completed',
      duration_ms: 4120,
      started_at: new Date(Date.now() - 12 * 60000).toISOString(),
    },
    {
      id: 'exec-003',
      workflow_id: 'support-routing',
      workflow_name: 'Support Ticket Router',
      status: 'failed',
      duration_ms: 1200,
      started_at: new Date(Date.now() - 15 * 60000).toISOString(),
    },
    {
      id: 'exec-004',
      workflow_id: 'research-pipeline',
      workflow_name: 'Research Pipeline',
      status: 'running',
      duration_ms: 0,
      started_at: new Date(Date.now() - 30000).toISOString(),
    },
  ]
}

function formatDuration(ms: number): string {
  if (ms === 0) return 'Running...'
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function formatTimeAgo(dateStr: string): string {
  const date = new Date(dateStr)
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000)
  
  if (seconds < 60) return 'just now'
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} hours ago`
  return `${Math.floor(seconds / 86400)} days ago`
}

const statusIcons = {
  completed: <CheckCircle className="h-5 w-5 text-green-500" />,
  failed: <XCircle className="h-5 w-5 text-red-500" />,
  running: <Play className="h-5 w-5 text-blue-500 animate-pulse" />,
}

export default function Dashboard() {
  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: fetchStats,
  })

  const { data: executions } = useQuery({
    queryKey: ['recentExecutions'],
    queryFn: fetchRecentExecutions,
    refetchInterval: 5000,
  })

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Overview of your AI Workflow Fabric
        </p>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-indigo-100 dark:bg-indigo-900/30 rounded-lg">
              <Bot className="h-6 w-6 text-indigo-600 dark:text-indigo-400" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Agents</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {stats?.agents ?? '-'}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
              <GitBranch className="h-6 w-6 text-purple-600 dark:text-purple-400" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Workflows</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {stats?.workflows ?? '-'}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded-lg">
              <Play className="h-6 w-6 text-green-600 dark:text-green-400" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Executions</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {stats?.executions ?? '-'}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-yellow-100 dark:bg-yellow-900/30 rounded-lg">
              <AlertTriangle className="h-6 w-6 text-yellow-600 dark:text-yellow-400" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Success Rate</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {stats?.successRate ? `${stats.successRate}%` : '-'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Quick actions */}
      <div className="flex flex-wrap gap-3">
        <Link
          to="/workflows/builder"
          className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors text-sm font-medium"
        >
          <GitBranch className="h-4 w-4" />
          New Workflow
        </Link>
        <Link
          to="/agents"
          className="inline-flex items-center gap-2 px-4 py-2 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors text-sm font-medium"
        >
          <Bot className="h-4 w-4" />
          Register Agent
        </Link>
      </div>

      {/* Recent executions */}
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Recent Executions</h2>
        </div>
        <div className="divide-y divide-gray-200 dark:divide-gray-700">
          {executions?.map((execution) => (
            <div key={execution.id} className="px-6 py-4 flex items-center justify-between">
              <div className="flex items-center gap-4">
                {statusIcons[execution.status]}
                <div>
                  <p className="font-medium text-gray-900 dark:text-white">
                    {execution.workflow_name}
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {execution.workflow_id}
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  {formatDuration(execution.duration_ms)}
                </p>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {formatTimeAgo(execution.started_at)}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
