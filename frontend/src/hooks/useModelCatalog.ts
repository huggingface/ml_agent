import { useState, useCallback, useEffect, useMemo } from "react";
import { apiFetch } from "@/utils/api";

export interface ModelOption {
  id: string;
  label: string;
  description: string;
  provider: string;
  providerLabel: string;
  avatarUrl?: string;
  recommended?: boolean;
  source?: string;
}

export interface ProviderOption {
  id: string;
  label: string;
  avatarUrl?: string;
  supportsCustomModel?: boolean;
  customModelHint?: string;
  customModelMode?: string;
  prefix?: string;
}

interface ModelCatalogResponse {
  current?: string;
  currentInfo?: ModelOption | null;
  available?: ModelOption[];
  providers?: ProviderOption[];
}

interface SessionResponse {
  model?: string;
}

const toModelOption = (value: unknown): ModelOption | null => {
  if (!value || typeof value !== "object") return null;
  const v = value as Record<string, unknown>;
  if (typeof v.id !== "string" || typeof v.label !== "string") return null;
  return {
    id: v.id,
    label: v.label,
    description: typeof v.description === "string" ? v.description : "",
    provider: typeof v.provider === "string" ? v.provider : "",
    providerLabel: typeof v.providerLabel === "string" ? v.providerLabel : "",
    avatarUrl: typeof v.avatarUrl === "string" ? v.avatarUrl : undefined,
    recommended: Boolean(v.recommended),
    source: typeof v.source === "string" ? v.source : undefined,
  };
};

const toProviderOption = (value: unknown): ProviderOption | null => {
  if (!value || typeof value !== "object") return null;
  const v = value as Record<string, unknown>;
  if (typeof v.id !== "string" || typeof v.label !== "string") return null;
  return {
    id: v.id,
    label: v.label,
    avatarUrl: typeof v.avatarUrl === "string" ? v.avatarUrl : undefined,
    supportsCustomModel: Boolean(v.supportsCustomModel),
    customModelHint:
      typeof v.customModelHint === "string" ? v.customModelHint : undefined,
    customModelMode:
      typeof v.customModelMode === "string" ? v.customModelMode : undefined,
    prefix: typeof v.prefix === "string" ? v.prefix : undefined,
  };
};

export function useModelCatalog(sessionId?: string) {
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([]);
  const [providerOptions, setProviderOptions] = useState<ProviderOption[]>([]);
  const [selectedModelPath, setSelectedModelPath] = useState<string>("");
  const [selectedModelInfo, setSelectedModelInfo] =
    useState<ModelOption | null>(null);

  useEffect(() => {
    let cancelled = false;

    const loadCatalog = async () => {
      try {
        const res = await apiFetch("/api/config/model");
        if (!res.ok || cancelled) return;
        const data = (await res.json()) as ModelCatalogResponse;
        const available = (data.available || [])
          .map(toModelOption)
          .filter((v): v is ModelOption => v !== null);
        const providers = (data.providers || [])
          .map(toProviderOption)
          .filter((v): v is ProviderOption => v !== null);
        const currentInfo = toModelOption(data.currentInfo ?? null);
        if (cancelled) return;

        setModelOptions(available);
        setProviderOptions(providers);
        setSelectedModelPath(data.current || "");
        setSelectedModelInfo(currentInfo);
      } catch {
        // ignore — catalog is optional, models may not be configured
      }
    };

    void loadCatalog();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    apiFetch(`/api/session/${sessionId}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data: SessionResponse | null) => {
        if (cancelled || !data?.model) return;
        setSelectedModelPath(data.model);
        const model = modelOptions.find((m) => m.id === data.model);
        if (model) {
          setSelectedModelInfo(model);
          return;
        }
        setSelectedModelInfo((prev) =>
          prev && prev.id === data.model
            ? prev
            : {
                id: data.model,
                label: data.model,
                description: "Custom model",
                provider: "",
                providerLabel: "",
              },
        );
      })
      .catch(() => {
        // ignore
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId, modelOptions]);

  const selectedModel =
    selectedModelInfo ||
    modelOptions.find((m) => m.id === selectedModelPath) ||
    (selectedModelPath
      ? {
          id: selectedModelPath,
          label: selectedModelPath,
          description: "Custom model",
          provider: "",
          providerLabel: "",
        }
      : null);

  const switchModel = useCallback(
    async (modelPath: string, info?: ModelOption) => {
      if (!sessionId) return;
      const res = await apiFetch(`/api/session/${sessionId}/model`, {
        method: "POST",
        body: JSON.stringify({ model: modelPath }),
      });
      if (res.ok) {
        setSelectedModelPath(modelPath);
        setSelectedModelInfo(
          info || modelOptions.find((m) => m.id === modelPath) || null,
        );
      }
    },
    [sessionId, modelOptions],
  );

  const groups = useMemo(
    () =>
      providerOptions.map((provider) => ({
        provider,
        models: modelOptions
          .filter((m) => m.provider === provider.id)
          .sort((a, b) => {
            const ar = a.recommended ? 0 : 1;
            const br = b.recommended ? 0 : 1;
            if (ar !== br) return ar - br;
            return a.label.localeCompare(b.label);
          }),
      })),
    [providerOptions, modelOptions],
  );

  return {
    modelOptions,
    providerOptions,
    selectedModelPath,
    selectedModel,
    switchModel,
    groups,
  };
}
