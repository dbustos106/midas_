// pnpm dev

import { McpServer, ResourceTemplate } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import express from 'express';
import fs from 'fs/promises';
import { z } from 'zod';
import path from 'path';
import url from 'url';

// Obtener __dirname en ESM
const __filename = url.fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Base segura desde donde serviremos ficheros (el directorio actual del script)
const localFilesBase = __dirname;

// Create an MCP server
const server = new McpServer({
    name: 'demo-server',
    version: '1.0.0'
});

// Add an addition tool
server.registerTool(
    'add',
    {
        title: 'Addition Tool',
        description: 'Add two numbers',
        inputSchema: {
            a: z.number(),
            b: z.number()
        },
        outputSchema: {
            result: z.number()
        },
    },
    async ({ a, b }) => {
        const output = { result: a + b };
        return {
            content: [{ type: 'text', text: JSON.stringify(output)}],
            structuredContent: output
        };
    }
);

// Get weather for a city
server.registerTool(
    'weather',
    {
        title: 'Weather Tool',
        description: 'Get weather for a city',
        inputSchema: {
            city: z.string().describe('name of a city')
        },
        outputSchema: {
            result: z.string()
        },
    },
    async ({ city }) => {
        const output = { result: `The weather in ${city} is soleado` };
        return {
            content: [
                { type: 'text', text: JSON.stringify(output) },
                {
                    type: 'resource_link',
                    uri: 'localfile://readme.md',
                    name: 'README.md',
                    mimeType: 'text/markdown',
                    description: 'A README file'
                },
            ],
            structuredContent: output
        };
    }
);

// Add a dynamic greeting resource
server.registerResource(
    'greeting',
    new ResourceTemplate('greeting://{name}', { list: undefined }),
    {
        title: 'Greeting Resource', // Display name for UI
        description: 'Dynamic greeting generator'
    },
    async (uri, { name }) => ({
        contents: [
            {
                uri: uri.href,
                text: `Hello, ${name}!`
            }
        ]
    })
);

// Registrar un ResourceTemplate para servir ficheros locales de forma controlada
server.registerResource(
    'localfile',
    new ResourceTemplate('localfile://{filename}', { list: undefined }),
    {
        title: 'Local file resource',
        description: 'Serve local files from the server directory via MCP'
    },
    // handler: recibe la URI y los parámetros extraídos del template ({ filename })
    async (uri, { filename } ) => {
        // Evitar traversal y normalizar el nombre: sólo permitimos servir archivos del directorio base
        const safeName = path.basename(filename as string);
        const filePath = path.join(localFilesBase, safeName);

        try {
            const content = await fs.readFile(filePath, 'utf8');
            return {
                contents: [
                    {
                        uri: uri.href, // p. ej. localfile://readme.md
                        text: content,
                        mimeType: 'text/markdown'
                    }
                ]
            };
        } catch (err: any) {
            return { contents: [] };
        }
    }
);

// Set up Express and HTTP transport
const app = express();
app.use(express.json());

app.post('/mcp', async (req, res) => {
    // Create a new transport for each request to prevent request ID collisions
    const transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: undefined,
        enableJsonResponse: true
    });

    res.on('close', () => {
        transport.close();
    });

    await server.connect(transport);
    await transport.handleRequest(req, res, req.body);
});

const port = parseInt(process.env.PORT || '3000');
app.listen(port, () => {
    console.log(`Demo MCP Server running on http://localhost:${port}/mcp`);
}).on('error', error => {
    console.error('Server error:', error);
    process.exit(1);
});
