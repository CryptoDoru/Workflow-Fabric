import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Bot, Plus, Trash2, Shield, RefreshCw } from 'lucide-react'
import { useState } from 'react'

interface Agent {
  id: string
  name: string
  version: string
  framework: string
  status: 'active' | 'inactive' | 'error'
  trust_score: number
  capabilities: string[]
  created_at: string
}

interface TrustScore {
  score: number
  sandbox_tier: string
  factors: {
    publisher_trust: number
    audit_status: number
    community_trust: number
    permission_analysis: number
    historical_behavior: number
  }
}

async function fetchAgents(): Promise<Agent[]> {
  try {
    const res = await fetch('/api/agents')
    if (res.ok) {
      const data = await res.json()
      return data.agents || []
    }
  } catch (e) {
    console.error('Failed to fetch agents:', e)
  }
  
  // Demo data
  return [
    {
      id: 'web-search-agent',
      name: 'Web Search Agent',
      version: '1.0.0',
      framework: 'langgraph',
      status: 'active',
      trust_score: 0.54,
      capabilities: ['web_search', 'summarize'],
      created_at: new Date().toISOString(),
    },
    {
      id: 'code-analysis-agent',
      name: 'Code Analysis Agent',
      version: '2.1.0',
      framework: 'crewai',
      status: 'active',
      trust_score: 0.67,
      capabilities: ['code_review', 'security_scan'],
      created_at: new Date().toISOString(),
    },
    {
      id: 'summarizer-agent',
      name: 'Summarizer Agent',
      version: '1.5.0',
      framework: 'autogen',
      status: 'active',
      trust_score: 0.72,
      capabilities: ['summarize', 'translate'],
      created_at: new Date().toISOString(),
    },
  ]
}

function getTrustColor(score: number): string {
  if (score >= 0.7) return 'text-green-600 dark:text-green-400'
  if (score >= 0.4) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-red-600 dark:text-red-400'
}

function getFrameworkColor(framework: string): string {
  switch (framework) {
    case 'langgraph':
      return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
    case 'crewai':
      return 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400'
    case 'autogen':
      return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
    default:
      return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400'
  }
}

export default function Agents() {
  const queryClient = useQueryClient()
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null)
  
  const { data: agents, isLoading, refetch } = useQuery({
    queryKey: ['agents'],
    queryFn: fetchAgents,
  })

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Agents</h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Manage your registered AI agents
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => refetch()}
            className="inline-flex items-center gap-2 px-4 py-2 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors text-sm font-medium"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </button>
          <button className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors text-sm font-medium">
            <Plus className="h-4 w-4" />
            Register Agent
          </button>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <RefreshCw className="h-8 w-8 text-gray-400 animate-spin" />
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {agents?.map((agent) => (
            <div
              key={agent.id}
              className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 hover:shadow-lg transition-shadow cursor-pointer"
              onClick={() => setSelectedAgent(selectedAgent === agent.id ? null : agent.id)}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg">
                    <Bot className="h-6 w-6 text-gray-600 dark:text-gray-400" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900 dark:text-white">{agent.name}</h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400">v{agent.version}</p>
                  </div>
                </div>
                <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getFrameworkColor(agent.framework)}`}>
                  {agent.framework}
                </span>
              </div>

              <div className="mt-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Shield className={`h-4 w-4 ${getTrustColor(agent.trust_score)}`} />
                  <span className={`text-sm font-medium ${getTrustColor(agent.trust_score)}`}>
                    {(agent.trust_score * 100).toFixed(0)}% Trust
                  </span>
                </div>
                <div className={`h-2 w-2 rounded-full ${
                  agent.status === 'active' ? 'bg-green-500' :
                  agent.status === 'inactive' ? 'bg-gray-400' : 'bg-red-500'
                }`} />
              </div>

              <div className="mt-4 flex flex-wrap gap-1">
                {agent.capabilities.slice(0, 3).map((cap) => (
                  <span
                    key={cap}
                    className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400"
                  >
                    {cap}
                  </span>
                ))}
                {agent.capabilities.length > 3 && (
                  <span className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400">
                    +{agent.capabilities.length - 3} more
                  </span>
                )}
              </div>

              {selectedAgent === agent.id && (
                <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    ID: <code className="text-xs bg-gray-100 dark:bg-gray-700 px-1 py-0.5 rounded">{agent.id}</code>
                  </p>
                  <div className="flex gap-2">
                    <button className="flex-1 px-3 py-2 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors">
                      View Details
                    </button>
                    <button className="px-3 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors">
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
