import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
    Send, Paperclip, FileText, Image as ImageIcon, Mic,
    X, Loader2, Bot, User, Brain, ChevronRight, File, Music, Folder, Plus, Trash2, FolderOpen
} from 'lucide-react'
import toast from 'react-hot-toast'

const CombinedQA = () => {
    const [input, setInput] = useState('')
    const [messages, setMessages] = useState([])
    const [isLoading, setIsLoading] = useState(false)
    const [files, setFiles] = useState({ documents: [], images: [], audio: [] })
    const [folders, setFolders] = useState([])
    const [selectedContext, setSelectedContext] = useState(null)
    const [showFileList, setShowFileList] = useState(true)
    const [renamingFolder, setRenamingFolder] = useState(null) // { id, name }
    const messagesEndRef = useRef(null)

    // Fetch all available files on mount
    useEffect(() => {
        fetchFiles()
    }, [])

    // Auto-scroll to bottom of chat
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    const fetchFiles = async () => {
        try {
            const [docsRes, imgsRes, audioRes, foldersRes] = await Promise.all([
                fetch('http://localhost:8000/api/documents'),
                fetch('http://localhost:8000/api/images'),
                fetch('http://localhost:8000/api/audio'),
                fetch('http://localhost:8000/api/folders')
            ])

            const docs = await docsRes.json()
            const imgs = await imgsRes.json()
            const audio = await audioRes.json()
            const foldersData = await foldersRes.json()

            // Tag files with their type for easier handling
            const allDocs = (docs.documents || []).map(f => ({ ...f, type: 'document' }))
            const allImgs = (imgs.images || []).map(f => ({ ...f, type: 'image' }))
            const allAudio = (audio.audio || []).map(f => ({ ...f, type: 'audio' }))

            setFiles({
                documents: allDocs,
                images: allImgs,
                audio: allAudio
            })
            setFolders(foldersData.folders || [])
        } catch (error) {
            console.error("Error fetching data:", error)
            toast.error("Failed to load content")
        }
    }

    const getFilesForCurrentContext = () => {
        if (selectedContext?.type === 'folder') {
            return {
                documents: files.documents.filter(f => f.folder_id === selectedContext.id),
                images: files.images.filter(f => f.folder_id === selectedContext.id),
                audio: files.audio.filter(f => f.folder_id === selectedContext.id)
            }
        }
        // "Unassigned" means no folder_id
        return {
            documents: files.documents.filter(f => !f.folder_id),
            images: files.images.filter(f => !f.folder_id),
            audio: files.audio.filter(f => !f.folder_id)
        }
    }

    const createFolder = async () => {
        const name = prompt("Enter folder name:")
        if (!name) return
        try {
            const res = await fetch('http://localhost:8000/api/folders', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name })
            })
            if (res.ok) {
                fetchFiles()
                toast.success("Folder created")
            }
        } catch (e) {
            toast.error("Failed to create folder")
        }
    }

    const deleteFolder = async (folderId, e) => {
        e.stopPropagation()
        if (!confirm("Delete this folder? Files will be unassigned.")) return
        try {
            await fetch(`http://localhost:8000/api/folders/${folderId}`, { method: 'DELETE' })
            fetchFiles()
            if (selectedContext?.id === folderId) setSelectedContext(null)
            toast.success("Folder deleted")
        } catch (e) {
            toast.error("Error deleting folder")
        }
    }

    const startRename = (folder, e) => {
        e.stopPropagation()
        setRenamingFolder({ id: folder.id, name: folder.name })
    }

    const saveRename = async (e) => {
        if (e.key === 'Enter') {
            try {
                const res = await fetch(`http://localhost:8000/api/folders/${renamingFolder.id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: renamingFolder.name })
                })
                if (res.ok) {
                    fetchFiles()
                    setRenamingFolder(null)
                    toast.success("Folder renamed")
                }
            } catch (err) {
                toast.error("Rename failed")
            }
        } else if (e.key === 'Escape') {
            setRenamingFolder(null)
        }
    }

    // --- Drag and Drop Logic ---

    const handleDragStart = (e, fileId, fileType) => {
        e.dataTransfer.setData("fileId", fileId)
        e.dataTransfer.setData("fileType", fileType)
    }

    const handleDragOver = (e) => {
        e.preventDefault() // Necessary to allow dropping
    }

    const handleDrop = async (e, folderId) => {
        e.preventDefault()
        const fileId = e.dataTransfer.getData("fileId")
        if (!fileId) return

        try {
            await fetch(`http://localhost:8000/api/folders/${folderId}/files`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_id: fileId })
            })
            toast.success("File moved to folder")
            fetchFiles() // Refresh counts
        } catch (err) {
            toast.error("Failed to move file")
        }
    }


    const handleSendMessage = async (e) => {
        e?.preventDefault()
        if (!input.trim() || isLoading) return

        const userMessage = input.trim()
        setInput('')
        setMessages(prev => [...prev, { role: 'user', content: userMessage }])
        setIsLoading(true)

        try {
            let endpoint = 'http://localhost:8000/api/question'
            let body = {
                question: userMessage,
                history: messages.slice(-10).map(m => ({
                    role: m.role,
                    content: m.content
                }))
            }

            if (selectedContext) {
                if (selectedContext.type === 'folder') {
                    body.folder_id = selectedContext.id
                } else {
                    body.file_id = selectedContext.id
                }
            }

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            })

            const data = await response.json()

            if (data.success) {
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: data.answer,
                    sources: data.sources
                }])
            } else {
                if (!selectedContext && data.error && data.error.includes("No relevant documents")) {
                    try {
                        const chatRes = await fetch('http://localhost:8000/api/chat', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                message: userMessage,
                                history: messages.slice(-10).map(m => ({
                                    role: m.role,
                                    content: m.content
                                }))
                            })
                        })
                        const chatData = await chatRes.json()
                        if (chatData.success) {
                            setMessages(prev => [...prev, {
                                role: 'assistant',
                                content: chatData.response,
                                sources: []
                            }])
                            return
                        }
                    } catch (fallbackErr) {
                        console.error("Fallback chat failed", fallbackErr)
                    }
                }
                throw new Error(data.error || 'Failed to get response')
            }
        } catch (error) {
            console.error('Error:', error)
            toast.error(error.message)
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: "I'm sorry, I encountered an error answering your question.",
                isError: true
            }])
        } finally {
            setIsLoading(false)
        }
    }

    const getFileIcon = (type) => {
        switch (type) {
            case 'image': return <ImageIcon className="w-4 h-4" />;
            case 'audio': return <Music className="w-4 h-4" />;
            default: return <FileText className="w-4 h-4" />;
        }
    }

    return (
        <div className="flex h-[calc(100vh-140px)] gap-4 font-sans text-gray-800">
            {/* File List Sidebar */}
            <motion.div
                initial={false}
                animate={{ width: showFileList ? 320 : 0, opacity: showFileList ? 1 : 0 }}
                className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden flex flex-col"
            >
                <div className="p-4 border-b border-gray-100 bg-gradient-to-r from-gray-50 to-white flex justify-between items-center min-w-[320px]">
                    <h3 className="font-semibold text-gray-700 flex items-center gap-2">
                        <FolderOpen className="w-5 h-5 text-indigo-500" /> Knowledge Base
                    </h3>
                    <div className="flex gap-2">
                        <button onClick={createFolder} className="p-1.5 hover:bg-indigo-50 hover:text-indigo-600 rounded-lg transition-colors" title="New Folder">
                            <Plus className="w-5 h-5" />
                        </button>
                        <button onClick={fetchFiles} className="text-xs text-indigo-600 hover:text-indigo-800 font-medium px-2 py-1 hover:bg-indigo-50 rounded-lg transition-colors">
                            Refresh
                        </button>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto p-3 min-w-[320px] space-y-4">
                    {/* Folders Section */}
                    {folders.length > 0 && (
                        <div>
                            <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider px-2 mb-2">Folders</h4>
                            <div className="space-y-1">
                                {folders.map(folder => (
                                    <div
                                        key={folder.id}
                                        onDragOver={handleDragOver}
                                        onDrop={(e) => handleDrop(e, folder.id)}
                                        onClick={() => setSelectedContext(selectedContext?.id === folder.id ? null : { type: 'folder', id: folder.id, name: folder.name })}
                                        className={`group flex items-center justify-between p-2.5 rounded-xl cursor-pointer text-sm transition-all duration-200 border ${selectedContext?.id === folder.id
                                            ? 'bg-indigo-50 text-indigo-700 border-indigo-200 shadow-sm'
                                            : 'border-transparent hover:bg-gray-50 text-gray-700 hover:border-gray-100'
                                            }`}
                                    >
                                        <div className="flex items-center gap-3 overflow-hidden">
                                            <Folder className={`w-5 h-5 flex-shrink-0 ${selectedContext?.id === folder.id ? 'fill-indigo-500 text-indigo-500' : 'fill-amber-100 text-amber-400'}`} />
                                            {renamingFolder && renamingFolder.id === folder.id ? (
                                                <input
                                                    autoFocus
                                                    className="bg-white border border-indigo-300 rounded px-2 py-1 w-full text-sm focus:outline-none focus:ring-2 focus:ring-indigo-200"
                                                    value={renamingFolder.name}
                                                    onChange={(e) => setRenamingFolder({ ...renamingFolder, name: e.target.value })}
                                                    onKeyDown={saveRename}
                                                    onClick={(e) => e.stopPropagation()}
                                                    onBlur={() => setRenamingFolder(null)}
                                                />
                                            ) : (
                                                <span className="font-medium truncate">{folder.name}</span>
                                            )}
                                        </div>
                                        <div className="flex items-center gap-1 opacity-100 sm:opacity-0 group-hover:opacity-100 transition-opacity">
                                            <span className="text-xs text-gray-400 font-mono bg-gray-100 px-1.5 py-0.5 rounded-md">{folder.file_count || 0}</span>
                                            <div className="flex bg-white rounded-lg border border-gray-100 shadow-sm opacity-0 group-hover:opacity-100 transition-all transform scale-95 group-hover:scale-100 ml-1">
                                                <button
                                                    onClick={(e) => startRename(folder, e)}
                                                    className="p-1.5 text-gray-400 hover:text-indigo-600 hover:bg-indigo-50 rounded-l-lg"
                                                    title="Rename"
                                                >
                                                    <FileText className="w-3 h-3" />
                                                </button>
                                                <div className="w-px bg-gray-100"></div>
                                                <button
                                                    onClick={(e) => deleteFolder(folder.id, e)}
                                                    className="p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-r-lg"
                                                    title="Delete"
                                                >
                                                    <Trash2 className="w-3 h-3" />
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    <div className="border-t border-gray-100 pt-4">
                        <div className="flex justify-between items-center px-2 mb-2">
                            <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider">
                                {selectedContext?.type === 'folder' ? `Files in ${selectedContext.name}` : 'Unassigned Files'}
                            </h4>
                            {(!selectedContext || selectedContext.type !== 'folder') && getFilesForCurrentContext().documents.length + getFilesForCurrentContext().images.length + getFilesForCurrentContext().audio.length > 0 && (
                                <button
                                    onClick={async (e) => {
                                        e.stopPropagation()
                                        if (!confirm("Are you sure you want to delete ALL unassigned files? This cannot be undone.")) return

                                        const unassigned = getFilesForCurrentContext()
                                        const allIds = [
                                            ...unassigned.documents.map(f => f.file_id),
                                            ...unassigned.images.map(f => f.file_id),
                                            ...unassigned.audio.map(f => f.file_id)
                                        ]

                                        if (allIds.length === 0) return

                                        try {
                                            const res = await fetch('http://localhost:8000/api/files/bulk-delete', {
                                                method: 'DELETE',
                                                headers: { 'Content-Type': 'application/json' },
                                                body: JSON.stringify({ file_ids: allIds })
                                            })
                                            const data = await res.json()
                                            if (data.success) {
                                                toast.success(`Deleted ${data.deleted_count} files`)
                                                fetchFiles()
                                            } else {
                                                toast.error("Failed to delete all files")
                                            }
                                        } catch (err) {
                                            console.error(err)
                                            toast.error("Error deleting files")
                                        }
                                    }}
                                    className="p-1 hover:bg-red-50 text-gray-400 hover:text-red-500 rounded transition-colors"
                                    title="Delete All Unassigned"
                                >
                                    <Trash2 className="w-3.5 h-3.5" />
                                </button>
                            )}
                        </div>
                        <p className="text-[10px] text-gray-400 px-2 mb-3">
                            {selectedContext?.type === 'folder'
                                ? "Files in this folder are used as context."
                                : "Drag files into folders to organize them."}
                        </p>

                        {/* Section: Documents */}
                        {getFilesForCurrentContext().documents.length > 0 && (
                            <div className="mb-4">
                                <h5 className="text-[10px] font-semibold text-gray-400 px-2 mb-1 flex items-center gap-1"><FileText className="w-3 h-3" /> Docs</h5>
                                <div className="space-y-1">
                                    {getFilesForCurrentContext().documents.map(doc => (
                                        <div
                                            key={doc.file_id}
                                            draggable
                                            onDragStart={(e) => handleDragStart(e, doc.file_id, 'document')}
                                            onClick={() => setSelectedContext(selectedContext?.id === doc.file_id ? null : { type: 'document', id: doc.file_id, name: doc.file_name })}
                                            className={`group/item w-full text-left p-2.5 rounded-xl text-sm flex items-center justify-between transition-colors cursor-move border ${selectedContext?.id === doc.file_id
                                                ? 'bg-blue-50 text-blue-700 border-blue-200'
                                                : 'border-transparent hover:bg-gray-50 text-gray-600 hover:border-gray-100'
                                                }`}
                                        >
                                            <div className="flex items-center gap-3 overflow-hidden">
                                                <FileText className="w-4 h-4 shrink-0 opacity-70" />
                                                <span className="truncate">{doc.file_name}</span>
                                            </div>
                                            {/* Delete Button (Only for Unassigned) */}
                                            {(!selectedContext || selectedContext.type !== 'folder') && (
                                                <button
                                                    onClick={async (e) => {
                                                        e.stopPropagation()
                                                        if (!confirm("Delete this file?")) return
                                                        try {
                                                            const res = await fetch('http://localhost:8000/api/files/bulk-delete', {
                                                                method: 'DELETE',
                                                                headers: { 'Content-Type': 'application/json' },
                                                                body: JSON.stringify({ file_ids: [doc.file_id] })
                                                            })
                                                            if ((await res.json()).success) {
                                                                toast.success("File deleted")
                                                                fetchFiles()
                                                                if (selectedContext?.id === doc.file_id) setSelectedContext(null)
                                                            }
                                                        } catch (err) { toast.error("Delete failed") }
                                                    }}
                                                    className="opacity-0 group-hover/item:opacity-100 p-1 hover:bg-white rounded text-gray-400 hover:text-red-500 transition-all"
                                                >
                                                    <Trash2 className="w-3 h-3" />
                                                </button>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Section: Images */}
                        {getFilesForCurrentContext().images.length > 0 && (
                            <div className="mb-4">
                                <h5 className="text-[10px] font-semibold text-gray-400 px-2 mb-1 flex items-center gap-1"><ImageIcon className="w-3 h-3" /> Images</h5>
                                <div className="space-y-1">
                                    {getFilesForCurrentContext().images.map(img => (
                                        <div
                                            key={img.file_id}
                                            draggable
                                            onDragStart={(e) => handleDragStart(e, img.file_id, 'image')}
                                            onClick={() => setSelectedContext(selectedContext?.id === img.file_id ? null : { type: 'image', id: img.file_id, name: img.file_name })}
                                            className={`group/item w-full text-left p-2.5 rounded-xl text-sm flex items-center justify-between transition-colors cursor-move border ${selectedContext?.id === img.file_id
                                                ? 'bg-purple-50 text-purple-700 border-purple-200'
                                                : 'border-transparent hover:bg-gray-50 text-gray-600 hover:border-gray-100'
                                                }`}
                                        >
                                            <div className="flex items-center gap-3 overflow-hidden">
                                                <ImageIcon className="w-4 h-4 shrink-0 opacity-70" />
                                                <span className="truncate">{img.file_name}</span>
                                            </div>
                                            {(!selectedContext || selectedContext.type !== 'folder') && (
                                                <button
                                                    onClick={async (e) => {
                                                        e.stopPropagation()
                                                        if (!confirm("Delete this file?")) return
                                                        try {
                                                            const res = await fetch('http://localhost:8000/api/files/bulk-delete', {
                                                                method: 'DELETE',
                                                                headers: { 'Content-Type': 'application/json' },
                                                                body: JSON.stringify({ file_ids: [img.file_id] })
                                                            })
                                                            if ((await res.json()).success) {
                                                                toast.success("File deleted")
                                                                fetchFiles()
                                                                if (selectedContext?.id === img.file_id) setSelectedContext(null)
                                                            }
                                                        } catch (err) { toast.error("Delete failed") }
                                                    }}
                                                    className="opacity-0 group-hover/item:opacity-100 p-1 hover:bg-white rounded text-gray-400 hover:text-red-500 transition-all"
                                                >
                                                    <Trash2 className="w-3 h-3" />
                                                </button>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Section: Audio */}
                        {getFilesForCurrentContext().audio.length > 0 && (
                            <div className="mb-4">
                                <h5 className="text-[10px] font-semibold text-gray-400 px-2 mb-1 flex items-center gap-1"><Music className="w-3 h-3" /> Audio</h5>
                                <div className="space-y-1">
                                    {getFilesForCurrentContext().audio.map(aud => (
                                        <div
                                            key={aud.file_id}
                                            draggable
                                            onDragStart={(e) => handleDragStart(e, aud.file_id, 'audio')}
                                            onClick={() => setSelectedContext(selectedContext?.id === aud.file_id ? null : { type: 'audio', id: aud.file_id, name: aud.file_name })}
                                            className={`group/item w-full text-left p-2.5 rounded-xl text-sm flex items-center justify-between transition-colors cursor-move border ${selectedContext?.id === aud.file_id
                                                ? 'bg-rose-50 text-rose-700 border-rose-200'
                                                : 'border-transparent hover:bg-gray-50 text-gray-600 hover:border-gray-100'
                                                }`}
                                        >
                                            <div className="flex items-center gap-3 overflow-hidden">
                                                <Music className="w-4 h-4 shrink-0 opacity-70" />
                                                <span className="truncate">{aud.file_name}</span>
                                            </div>
                                            {(!selectedContext || selectedContext.type !== 'folder') && (
                                                <button
                                                    onClick={async (e) => {
                                                        e.stopPropagation()
                                                        if (!confirm("Delete this file?")) return
                                                        try {
                                                            const res = await fetch('http://localhost:8000/api/files/bulk-delete', {
                                                                method: 'DELETE',
                                                                headers: { 'Content-Type': 'application/json' },
                                                                body: JSON.stringify({ file_ids: [aud.file_id] })
                                                            })
                                                            if ((await res.json()).success) {
                                                                toast.success("File deleted")
                                                                fetchFiles()
                                                                if (selectedContext?.id === aud.file_id) setSelectedContext(null)
                                                            }
                                                        } catch (err) { toast.error("Delete failed") }
                                                    }}
                                                    className="opacity-0 group-hover/item:opacity-100 p-1 hover:bg-white rounded text-gray-400 hover:text-red-500 transition-all"
                                                >
                                                    <Trash2 className="w-3 h-3" />
                                                </button>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {Object.values(getFilesForCurrentContext()).every(arr => arr.length === 0) && (
                            <div className="text-center text-gray-400 py-12 text-sm flex flex-col items-center">
                                <div className="w-12 h-12 bg-gray-50 rounded-full flex items-center justify-center mb-3">
                                    <File className="w-6 h-6 text-gray-300" />
                                </div>
                                <p>No files found.</p>
                                <p className="text-xs">Upload some content in the Upload tab!</p>
                            </div>
                        )}
                    </div>
                </div>
            </motion.div>

            {/* Toggle Sidebar Button (visible when closed) */}
            {!showFileList && (
                <button
                    onClick={() => setShowFileList(true)}
                    className="absolute left-4 top-1/2 bg-white p-2.5 rounded-full shadow-lg border border-gray-100 hover:bg-indigo-50 hover:text-indigo-600 z-10 transition-all"
                >
                    <ChevronRight className="w-5 h-5" />
                </button>
            )}

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden relative">
                <div className="p-4 border-b border-gray-100 bg-white/80 backdrop-blur-sm flex justify-between items-center z-10">
                    <div className="flex items-center gap-3">
                        {selectedContext ? (
                            <motion.div
                                initial={{ opacity: 0, y: -10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="flex items-center gap-3"
                            >
                                <span className={`p-2 rounded-lg shadow-sm border ${selectedContext.type === 'folder'
                                    ? 'bg-indigo-50 border-indigo-100 text-indigo-600'
                                    : 'bg-white border-gray-100 text-gray-600'
                                    }`}>
                                    {selectedContext.type === 'folder' ? <Folder className="w-5 h-5" /> : getFileIcon(selectedContext.type)}
                                </span>
                                <div>
                                    <p className="text-xs text-indigo-500 font-bold uppercase tracking-wider">{selectedContext.type === 'folder' ? 'Folder Context' : 'Chatting with'}</p>
                                    <p className="text-base font-bold text-gray-900">{selectedContext.name}</p>
                                </div>
                            </motion.div>
                        ) : (
                            <div className="flex items-center gap-3">
                                <span className="p-2 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg shadow-md text-white">
                                    <Bot className="w-5 h-5" />
                                </span>
                                <div>
                                    <p className="text-base font-bold text-gray-900">General Chat</p>
                                    <p className="text-xs text-gray-400">Searching across all files</p>
                                </div>
                            </div>
                        )}
                    </div>

                    {showFileList && (
                        <button onClick={() => setShowFileList(false)} className="p-1.5 hover:bg-gray-100 rounded-lg text-gray-400 hover:text-gray-600 transition-colors">
                            <X className="w-5 h-5" />
                        </button>
                    )}
                </div>

                <div className="flex-1 overflow-y-auto p-6 space-y-8 bg-gradient-to-br from-gray-50/50 to-white/50">
                    {messages.length === 0 && (
                        <div className="h-full flex flex-col items-center justify-center text-center opacity-40 select-none">
                            <div className="w-24 h-24 bg-gradient-to-tr from-gray-200 to-gray-100 rounded-full flex items-center justify-center mb-6">
                                <Brain className="w-12 h-12 text-gray-400" />
                            </div>
                            <h3 className="text-2xl font-bold text-gray-700 mb-2">Prism Unified QA</h3>
                            <p className="max-w-md text-gray-500">
                                Drag files into folders to organize your knowledge base. Select a context to start chatting.
                            </p>
                        </div>
                    )}

                    {messages.map((msg, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            {msg.role === 'assistant' && (
                                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-600 to-purple-600 flex items-center justify-center shrink-0 shadow-md text-white mt-1">
                                    <Bot className="w-4 h-4" />
                                </div>
                            )}

                            <div className={`max-w-[80%] space-y-2`}>
                                <div className={`p-5 rounded-2xl shadow-sm leading-relaxed text-sm ${msg.role === 'user'
                                    ? 'bg-indigo-600 text-white rounded-br-sm'
                                    : 'bg-white text-gray-800 border border-gray-100 rounded-bl-sm shadow-md'
                                    }`}>
                                    {msg.content}
                                </div>

                                {/* Sources/Metadata */}
                                {msg.sources && msg.sources.length > 0 && (
                                    <div className="flex gap-2 flex-wrap ml-1">
                                        {msg.sources.map((source, sIdx) => (
                                            <span key={sIdx} className="text-xs bg-white text-gray-500 px-2 py-1 rounded-full border border-gray-200 flex items-center gap-1.5 hover:bg-gray-50 transition-colors cursor-default shadow-sm">
                                                <File className="w-3 h-3 text-indigo-400" />
                                                <span className="font-medium">{source.file_name}</span>
                                                {source.page !== undefined && <span className="text-gray-400 text-[10px]">p.{source.page + 1}</span>}
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </div>

                            {msg.role === 'user' && (
                                <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center shrink-0 text-gray-500 mt-1">
                                    <User className="w-4 h-4" />
                                </div>
                            )}
                        </motion.div>
                    ))}
                    {isLoading && (
                        <div className="flex gap-4">
                            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-600 to-purple-600 flex items-center justify-center shrink-0 shadow-md text-white mt-1">
                                <Bot className="w-4 h-4" />
                            </div>
                            <div className="bg-white p-4 rounded-2xl rounded-bl-sm border border-gray-100 flex items-center gap-3 shadow-md">
                                <Loader2 className="w-4 h-4 animate-spin text-indigo-500" />
                                <span className="text-sm text-gray-500 font-medium">Processing...</span>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                <form onSubmit={handleSendMessage} className="p-4 bg-white border-t border-gray-100">
                    <div className="relative flex items-center gap-3 max-w-4xl mx-auto">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder={selectedContext ? `Message ${selectedContext.name}...` : "Message General Chat..."}
                            className="flex-1 bg-gray-50 border border-gray-200 rounded-xl px-5 py-3.5 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all pr-12 shadow-inner text-sm"
                            disabled={isLoading}
                        />
                        <button
                            type="submit"
                            disabled={!input.trim() || isLoading}
                            className="absolute right-2 p-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md transform hover:scale-105 active:scale-95"
                        >
                            <Send className="w-4 h-4" />
                        </button>
                    </div>
                    <div className="text-center mt-2">
                        <p className="text-[10px] text-gray-400">Prism AI can make mistakes. Verify important information.</p>
                    </div>
                </form>
            </div>
        </div>
    )
}

export default CombinedQA
