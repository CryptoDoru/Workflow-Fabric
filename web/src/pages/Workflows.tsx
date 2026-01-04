import { useQuery } from '@tanstack/react-query'
import { GitBranch, Plus, Play, Clock, CheckCircle, XCircle } from 'lucide-react'
import { Link } from 'react-router-dom'

interface Workflow {
  id: string
  name: string
  description?: string
  version: string
  steps: number
  last_execution?: {
    status: 'completed' | 'failed' | 'running'
    duration_ms: number
    timestamp: string
  }
  created_at: string
}

async function fetchWorkflows(): Promise<Workflow[]> {
  try {
    const res = await fetch('/api/workflows')
    if (res.ok) {
      const data = await res.json()
      return data.workflows || []
    }
  } catch (e) {
    console.error('Failed to fetch workflows:', e)
  }
  
  return [
    {
      id: 'research-pipeline',
      name: 'Research Pipeline',
      description: 'Multi-agent research workflow: search → analyze → summarize',
      version: '1.0.0',
      steps: 3,
      last_execution: {
        status: 'completed',
        duration_ms: 2340,
        timestamp: new Date(Date.now() - 5 * 60000).toISOString(),
      },
      created_at: new Date(Date.now() - 7 * 24 * 60 * 60000).toISOString(),
    },
    {
      id: 'code-review',
      name: 'Code Review Pipeline',
      description: 'Automated code review: parse → security scan → quality check → report',
      version: '2.0.0',
      steps: 4,
      last_execution: {
        status: 'completed',
        duration_ms: 4120,
        timestamp: new Date(Date.now() - 12 * 60000).toISOString(),
      },
      created_at: new Date(Date.now() - 14 * 24 * 60 * 60000).toISOString(),
    },
    {
      id: 'support-routing',
      name: 'Support Ticket Router',
      description: 'Classify and route support tickets to specialist agents',
      version: '1.2.0',
      steps: 3,
      last_execution: {
        status: 'failed',
        duration_ms: 1200,
        timestamp: new Date(Date.now() - 15 * 60000).toISOString(),
      },
      created_at: new Date(Date.now() - 30 * 24 * 60 * 60000).toISOString(),
    },
  ]
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString()
}

const statusConfig = {
  completed: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-100 dark:bg-green-900/30' },
  failed: { icon: XCircle, color: 'text-red-500', bg: 'bg-red-100 dark:bg-red-900/30' },
  running: { icon: Clock, color: 'text-blue-500 animate-pulse', bg: 'bg-blue-100 dark:bg-blue-900/30' },
}

export default function Workflows() {
  const { data: workflows, isLoading } = useQuery({
    queryKey: ['workflows'],
    queryFn: fetchWorkflows,
  })

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Workflows</h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Create and manage multi-agent workflows
          </p>
        </div>
        <Link
          to="/workflows/builder"
          className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors text-sm font-medium"
        >
          <Plus className="h-4 w-4" />
          New Workflow
        </Link>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-900">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Workflow
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Steps
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Last Run
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Created
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
            {workflows?.map((workflow) => {
              const StatusIcon = workflow.last_execution 
                ? statusConfig[workflow.last_execution.status].icon 
                : Clock
              const statusColor = workflow.last_execution 
                ? statusConfig[workflow.last_execution.status].color 
                : 'text-gray-400'
              
              return (
                <tr key={workflow.id} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
                        <GitBranch className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                      </div>
                      <div>
                        <p className="font-medium text-gray-900 dark:text-white">{workflow.name}</p>
                        <p className="text-sm text-gray-500 dark:text-gray-400">{workflow.description}</p>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
                      {workflow.steps} steps
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    {workflow.last_execution ? (
                      <div className="flex items-center gap-2">
                        <StatusIcon className={`h-4 w-4 ${statusColor}`} />
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                          {formatDuration(workflow.last_execution.duration_ms)}
                        </span>
                      </div>
                    ) : (
                      <span className="text-sm text-gray-400">Never</span>
                    )}
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
                    {formatDate(workflow.created_at)}
                  </td>
                  <td className="px-6 py-4 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <button className="p-2 text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">
                        <Play className="h-4 w-4" />
                      </button>
                      <Link
                        to={`/workflows/builder/${workflow.id}`}
                        className="p-2 text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                      >
                        <GitBranch className="h-4 w-4" />
                      </Link>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
