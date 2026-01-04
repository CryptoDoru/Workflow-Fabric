import { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { Bot } from 'lucide-react'

interface AgentNodeData {
  label: string
  agentId?: string
  timeout?: number
  retry?: {
    maxAttempts: number
    backoffMs: number
  }
}

function AgentNode({ data, selected }: NodeProps<AgentNodeData>) {
  return (
    <div
      className={`px-4 py-3 shadow-lg rounded-lg border-2 bg-white dark:bg-gray-800 min-w-[150px] ${
        selected
          ? 'border-indigo-500 ring-2 ring-indigo-200 dark:ring-indigo-800'
          : 'border-gray-200 dark:border-gray-700'
      }`}
    >
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 !bg-gray-400 dark:!bg-gray-500"
      />
      
      <div className="flex items-center gap-2">
        <div className="p-1.5 bg-indigo-100 dark:bg-indigo-900/30 rounded">
          <Bot className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
        </div>
        <div>
          <p className="font-medium text-sm text-gray-900 dark:text-white">
            {data.label}
          </p>
          {data.agentId && (
            <p className="text-xs text-gray-500 dark:text-gray-400">
              {data.agentId}
            </p>
          )}
        </div>
      </div>
      
      {data.timeout && (
        <div className="mt-2 text-xs text-gray-400 dark:text-gray-500">
          Timeout: {data.timeout / 1000}s
        </div>
      )}
      
      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 !bg-indigo-500"
      />
    </div>
  )
}

export default memo(AgentNode)
