import { useEffect, useState } from "react";
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  TextField,
  Typography,
} from "@mui/material";

interface CustomModelDialogProps {
  open: boolean;
  prefix: string;
  onClose: () => void;
  onSubmit: (modelId: string) => Promise<void>;
}

export default function CustomModelDialog({
  open,
  prefix,
  onClose,
  onSubmit,
}: CustomModelDialogProps) {
  const [value, setValue] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!open) {
      setValue("");
      setError("");
      setSubmitting(false);
    }
  }, [open]);

  const handleSubmit = async () => {
    const trimmed = value.trim();
    if (!trimmed) {
      setError("Model id is required");
      return;
    }
    setError("");
    setSubmitting(true);
    try {
      await onSubmit(trimmed);
    } catch {
      setError("Failed to switch model");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog
      open={open}
      onClose={submitting ? undefined : onClose}
      slotProps={{
        backdrop: {
          sx: {
            backgroundColor: "rgba(0,0,0,0.5)",
            backdropFilter: "blur(4px)",
          },
        },
      }}
      PaperProps={{
        sx: {
          bgcolor: "var(--panel)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-md)",
          boxShadow: "var(--shadow-1)",
          maxWidth: 520,
          mx: 2,
          width: "100%",
        },
      }}
    >
      <DialogTitle
        sx={{
          color: "var(--text)",
          fontWeight: 700,
          fontSize: "1rem",
          pt: 2.5,
          pb: 1,
          px: 3,
        }}
      >
        Custom OpenAI-compatible model
      </DialogTitle>
      <DialogContent sx={{ px: 3, pt: 0.5, pb: 0 }}>
        <Typography
          sx={{ color: "var(--muted-text)", fontSize: "0.85rem", mb: 1.5 }}
        >
          Enter model id only. We will use server env config for base URL and
          key.
        </Typography>
        <TextField
          fullWidth
          autoFocus
          size="small"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !submitting) void handleSubmit();
          }}
          placeholder="e.g. my-model"
          disabled={submitting}
          sx={{
            "& .MuiOutlinedInput-root": {
              bgcolor: "transparent",
              color: "var(--text)",
            },
          }}
        />
        <Typography
          sx={{ color: "var(--muted-text)", fontSize: "0.78rem", mt: 1 }}
        >
          Final id:{" "}
          <code>
            {prefix}
            {value.trim() || "<model-id>"}
          </code>
        </Typography>
        {error && (
          <Typography
            sx={{ color: "var(--accent-red)", fontSize: "0.78rem", mt: 1 }}
          >
            {error}
          </Typography>
        )}
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2.5, pt: 2, gap: 1 }}>
        <Button
          onClick={onClose}
          disabled={submitting}
          size="small"
          sx={{
            color: "var(--muted-text)",
            fontSize: "0.82rem",
            px: 2,
            textTransform: "none",
            "&:hover": { bgcolor: "var(--hover-bg)" },
          }}
        >
          Cancel
        </Button>
        <Button
          onClick={() => void handleSubmit()}
          disabled={submitting}
          variant="contained"
          size="small"
          sx={{
            fontSize: "0.82rem",
            px: 2.5,
            bgcolor: "var(--accent-yellow)",
            color: "#000",
            textTransform: "none",
            fontWeight: 700,
            boxShadow: "none",
            "&:hover": { bgcolor: "#FFB340", boxShadow: "none" },
          }}
        >
          {submitting ? "Switching…" : "Switch model"}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
