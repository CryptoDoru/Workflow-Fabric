import { useQuery } from '@tanstack/react-query'
import { Play, CheckCircle, XCircle, Clock, RefreshCw, Filter, Wifi, WifiOff } from 'lucide-react'
import { useState, useEffect, useCallback } from 'react'

interface Execution {
  id: string
  workflow_id: string
  workflow_name: string
  status: 'completed' | 'failed' | 'running' | 'pending'
  duration_ms: number
  steps_completed: number
  total_steps: number
  started_at: string
  completed_at?: string
  error?: string
}

interface ExecutionEvent {
  type: string
  execution_id: string
  step_id?: string
  status?: string
  output?: any
  error?: string
  timestamp: string
}

async function fetchExecutions(): Promise<Execution[]> {
  try {
    const res = await fetch('/api/executions')
    if (res.ok) {
      const data = await res.json()
      return data.executions || []
    }
  } catch (e) {
    console.error('Failed to fetch executions:', e)
  }
  
  return [
    {
      id: 'exec-001',
      workflow_id: 'research-pipeline',
      workflow_name: 'Research Pipeline',
      status: 'completed',
      duration_ms: 2340,
      steps_completed: 3,
      total_steps: 3,
      started_at: new Date(Date.now() - 5 * 60000).toISOString(),
      completed_at: new Date(Date.now() - 5 * 60000 + 2340).toISOString(),
    },
    {
      id: 'exec-002',
      workflow_id: 'code-review',
      workflow_name: 'Code Review Pipeline',
      status: 'completed',
      duration_ms: 4120,
      steps_completed: 4,
      total_steps: 4,
      started_at: new Date(Date.now() - 12 * 60000).toISOString(),
      completed_at: new Date(Date.now() - 12 * 60000 + 4120).toISOString(),
    },
    {
      id: 'exec-003',
      workflow_id: 'support-routing',
      workflow_name: 'Support Ticket Router',
      status: 'failed',
      duration_ms: 1200,
      steps_completed: 1,
      total_steps: 3,
      started_at: new Date(Date.now() - 15 * 60000).toISOString(),
      completed_at: new Date(Date.now() - 15 * 60000 + 1200).toISOString(),
      error: 'Agent timeout: ticket-classifier-agent did not respond within 30s',
    },
    {
      id: 'exec-004',
      workflow_id: 'research-pipeline',
      workflow_name: 'Research Pipeline',
      status: 'running',
      duration_ms: 0,
      steps_completed: 1,
      total_steps: 3,
      started_at: new Date(Date.now() - 30000).toISOString(),
    },
    {
      id: 'exec-005',
      workflow_id: 'code-review',
      workflow_name: 'Code Review Pipeline',
      status: 'pending',
      duration_ms: 0,
      steps_completed: 0,
      total_steps: 4,
      started_at: new Date(Date.now() - 5000).toISOString(),
    },
  ]
}

function formatDuration(ms: number): string {
  if (ms === 0) return '-'
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`
}

function formatTime(dateStr: string): string {
  return new Date(dateStr).toLocaleTimeString()
}

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString()
}

const statusConfig = {
  completed: { 
    icon: CheckCircle, 
    color: 'text-green-500', 
    bg: 'bg-green-100 dark:bg-green-900/30',
    label: 'Completed'
  },
  failed: { 
    icon: XCircle, 
    color: 'text-red-500', 
    bg: 'bg-red-100 dark:bg-red-900/30',
    label: 'Failed'
  },
  running: { 
    icon: RefreshCw, 
    color: 'text-blue-500', 
    bg: 'bg-blue-100 dark:bg-blue-900/30',
    label: 'Running'
  },
  pending: { 
    icon: Clock, 
    color: 'text-yellow-500', 
    bg: 'bg-yellow-100 dark:bg-yellow-900/30',
    label: 'Pending'
  },
}

// Custom hook for WebSocket connection with auto-reconnect
function useWebSocket(url: string, onMessage: (event: ExecutionEvent) => void) {
  const [isConnected, setIsConnected] = useState(false)
  const [ws, setWs] = useState<WebSocket | null>(null)

  const connect = useCallback(() => {
    try {
      const socket = new WebSocket(url)
      
      socket.onopen = () => {
        setIsConnected(true)
        console.log('WebSocket connected')
      }
      
      socket.onclose = () => {
        setIsConnected(false)
        console.log('WebSocket disconnected, reconnecting in 3s...')
        setTimeout(connect, 3000)
      }
      
      socket.onerror = (error) => {
        console.error('WebSocket error:', error)
      }
      
      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          onMessage(data)
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e)
        }
      }
      
      setWs(socket)
    } catch (e) {
      console.error('Failed to connect WebSocket:', e)
      setTimeout(connect, 3000)
    }
  }, [url, onMessage])

  useEffect(() => {
    connect()
    return () => {
      ws?.close()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return { isConnected, ws }
}

export default function Executions() {
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [realtimeEvents, setRealtimeEvents] = useState<ExecutionEvent[]>([])
  
  const { data: executions, isLoading, refetch } = useQuery({
    queryKey: ['executions'],
    queryFn: fetchExecutions,
    refetchInterval: 5000,
  })

  // Handle incoming WebSocket events
  const handleWebSocketMessage = useCallback((event: ExecutionEvent) => {
    setRealtimeEvents(prev => [event, ...prev].slice(0, 50)) // Keep last 50 events
    
    // Trigger refetch when execution status changes
    if (event.type === 'workflow_completed' || event.type === 'workflow_failed' || event.type === 'step_completed') {
      refetch()
    }
  }, [refetch])

  // Connect to WebSocket for all events
  const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/events`
  const { isConnected } = useWebSocket(wsUrl, handleWebSocketMessage)

  const filteredExecutions = executions?.filter(e => 
    statusFilter === 'all' || e.status === statusFilter
  )

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Executions</h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Monitor workflow execution history
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* WebSocket connection status */}
          <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium ${
            isConnected 
              ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' 
              : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
          }`}>
            {isConnected ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
            {isConnected ? 'Live' : 'Disconnected'}
          </div>
          
          <div className="flex items-center gap-2">
            <Filter className="h-4 w-4 text-gray-400" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="text-sm border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
            >
              <option value="all">All Status</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
              <option value="running">Running</option>
              <option value="pending">Pending</option>
            </select>
          </div>
          <button
            onClick={() => refetch()}
            className="inline-flex items-center gap-2 px-4 py-2 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors text-sm font-medium"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* Real-time events feed */}
      {realtimeEvents.length > 0 && (
        <div className="bg-gray-50 dark:bg-gray-800/50 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
            </span>
            Live Events
          </h3>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {realtimeEvents.slice(0, 5).map((event, i) => (
              <div key={i} className="flex items-center gap-3 text-sm">
                <span className="text-gray-400 dark:text-gray-500 text-xs">
                  {new Date(event.timestamp).toLocaleTimeString()}
                </span>
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                  event.type.includes('completed') ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' :
                  event.type.includes('failed') ? 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400' :
                  'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400'
                }`}>
                  {event.type}
                </span>
                <span className="text-gray-600 dark:text-gray-400 truncate">
                  {event.execution_id}
                  {event.step_id && ` / ${event.step_id}`}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="space-y-4">
        {isLoading ? (
          <div className="text-center py-12 text-gray-500 dark:text-gray-400">
            Loading executions...
          </div>
        ) : filteredExecutions?.length === 0 ? (
          <div className="text-center py-12 text-gray-500 dark:text-gray-400">
            No executions found
          </div>
        ) : (
          filteredExecutions?.map((execution) => {
            const config = statusConfig[execution.status]
            const StatusIcon = config.icon
            const progress = (execution.steps_completed / execution.total_steps) * 100

            return (
              <div
                key={execution.id}
                className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6"
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-4">
                    <div className={`p-3 rounded-lg ${config.bg}`}>
                      <StatusIcon className={`h-5 w-5 ${config.color} ${execution.status === 'running' ? 'animate-spin' : ''}`} />
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900 dark:text-white">
                        {execution.workflow_name}
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        ID: {execution.id}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${config.bg} ${config.color}`}>
                      {config.label}
                    </span>
                    <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                      {formatDate(execution.started_at)} {formatTime(execution.started_at)}
                    </p>
                  </div>
                </div>

                <div className="mt-4">
                  <div className="flex items-center justify-between text-sm mb-2">
                    <span className="text-gray-600 dark:text-gray-400">
                      Steps: {execution.steps_completed} / {execution.total_steps}
                    </span>
                    <span className="text-gray-600 dark:text-gray-400">
                      Duration: {formatDuration(execution.duration_ms)}
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all ${
                        execution.status === 'failed' ? 'bg-red-500' :
                        execution.status === 'completed' ? 'bg-green-500' :
                        'bg-blue-500'
                      }`}
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>

                {execution.error && (
                  <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                    <p className="text-sm text-red-700 dark:text-red-400">
                      {execution.error}
                    </p>
                  </div>
                )}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
