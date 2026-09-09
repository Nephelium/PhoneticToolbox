/** Generated from contracts/openapi.json. Do not edit. */
export interface paths {
    "/api/v1/assets": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Listing */
        get: operations["list_assets"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/assets/{asset_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Metadata */
        get: operations["get_asset"];
        put?: never;
        post?: never;
        /** Delete */
        delete: operations["delete_asset"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/assets/{asset_id}/content": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Download */
        get: operations["download_asset"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/assets/{asset_id}/delete-impact": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Impact */
        get: operations["get_delete_impact"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/auth/challenge": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Challenge */
        get: operations["get_login_challenge"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/auth/login": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Login */
        post: operations["login_account"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/auth/logout": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Logout */
        post: operations["logout_account"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/auth/me": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Me */
        get: operations["get_current_account"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/capabilities": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Capabilities */
        get: operations["get_capabilities"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/health": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Health */
        get: operations["get_health"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Listing */
        get: operations["list_jobs"];
        put?: never;
        /** Create */
        post: operations["create_job"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/{job_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get */
        get: operations["get_job"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/{job_id}/cancel": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Cancel */
        post: operations["cancel_job"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/{job_id}/events": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Events */
        get: operations["get_job_events"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/{job_id}/retry": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Retry */
        post: operations["retry_job"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/projects": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Listing */
        get: operations["list_projects"];
        put?: never;
        /** Create */
        post: operations["create_project"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/projects/{project_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get */
        get: operations["get_project"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        /** Rename */
        patch: operations["rename_project"];
        trace?: never;
    };
    "/api/v1/storage/usage": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Usage */
        get: operations["get_storage_usage"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/uploads": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Create */
        post: operations["create_upload"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/uploads/{asset_id}/blocks": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        /** Append */
        put: operations["append_upload_block"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/uploads/{asset_id}/finalize": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Finalize */
        post: operations["finalize_upload"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
}
export type webhooks = Record<string, never>;
export interface components {
    schemas: {
        /** AcousticAssetRef */
        AcousticAssetRef: {
            /** Asset Id */
            asset_id: string;
            /** Sha256 */
            sha256: string;
        };
        /** AcousticBackendObservation */
        AcousticBackendObservation: {
            /**
             * Actual
             * @enum {string}
             */
            actual: "native_reaper" | "reaper_python" | "irapt1" | "praat_fallback" | "unavailable" | "disabled";
            /**
             * Reason
             * @default null
             */
            reason: string | null;
            /**
             * Resource Sha256
             * @default null
             */
            resource_sha256: string | null;
            /**
             * Stage
             * @enum {string}
             */
            stage: "reaper" | "wm_f0";
        };
        /** AcousticBackendPolicy */
        AcousticBackendPolicy: {
            /**
             * Reaper
             * @default native_then_python
             * @enum {string}
             */
            reaper: "native_required" | "native_then_python" | "python_only" | "disabled";
            /**
             * Wm F0
             * @default irapt_then_praat
             * @constant
             */
            wm_f0: "irapt_then_praat";
        };
        /** AcousticBatchCounts */
        AcousticBatchCounts: {
            /** Cancel Requested */
            cancel_requested: number;
            /** Cancelled */
            cancelled: number;
            /** Failed */
            failed: number;
            /** Interrupted */
            interrupted: number;
            /** Not Started */
            not_started: number;
            /** Queued */
            queued: number;
            /** Running */
            running: number;
            /** Succeeded */
            succeeded: number;
        };
        /** AcousticBatchItem */
        AcousticBatchItem: {
            /** Audio Asset Id */
            audio_asset_id: string;
            /**
             * Error Code
             * @default null
             */
            error_code: string | null;
            /** Index */
            index: number;
            /** Job Id */
            job_id: string | null;
            /**
             * State
             * @enum {string}
             */
            state: "not_started" | "queued" | "running" | "cancel_requested" | "succeeded" | "failed" | "cancelled" | "interrupted";
        };
        /** AcousticBatchSummary */
        AcousticBatchSummary: {
            /** Batch Id */
            batch_id: string;
            /** Closed */
            closed: boolean;
            /**
             * Complete
             * @description True only when the closed batch succeeded for every requested file
             */
            complete: boolean;
            counts: components["schemas"]["AcousticBatchCounts"];
            /** Items */
            items: components["schemas"]["AcousticBatchItem"][];
            /**
             * Kind
             * @default acoustic_batch
             * @constant
             */
            kind: "acoustic_batch";
            /**
             * Schema Version
             * @default m01/1
             * @constant
             */
            schema_version: "m01/1";
            /** Total */
            total: number;
        };
        /** AcousticConfigSnapshot */
        AcousticConfigSnapshot: {
            backend_policy?: components["schemas"]["AcousticBackendPolicy"];
            selection?: components["schemas"]["AcousticSelection"];
            settings?: components["schemas"]["AcousticSettings"];
        };
        /** AcousticDecodedAudio */
        AcousticDecodedAudio: {
            /** Channels */
            channels: number;
            /**
             * Sample Count
             * @description Frames per channel, never the flattened channel sample count
             */
            sample_count: number;
            /**
             * Sample Dtype
             * @enum {string}
             */
            sample_dtype: "uint8" | "int16" | "int32" | "float32" | "float64";
            /** Sample Rate Hz */
            sample_rate_hz: number;
        };
        /** AcousticFileManifest */
        AcousticFileManifest: {
            /**
             * Complete
             * @default true
             * @constant
             */
            complete: true;
            /** Completed At */
            completed_at: number;
            /** Expires At */
            expires_at: number | null;
            /** Files */
            files: components["schemas"]["AcousticResultFile"][];
            /** Job Id */
            job_id: string;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "acoustic_file";
            metadata: components["schemas"]["AcousticMetadata"];
            /**
             * Retention
             * @enum {string}
             */
            retention: "server" | "local";
            /** Row Count */
            row_count: number;
            /**
             * Schema Version
             * @default m01/1
             * @constant
             */
            schema_version: "m01/1";
        };
        /** AcousticInputSnapshot */
        AcousticInputSnapshot: {
            /** Asset Id */
            asset_id: string;
            /** Expires At */
            expires_at: number | null;
            /**
             * Role
             * @enum {string}
             */
            role: "audio" | "textgrid" | "lip";
            /** Sha256 */
            sha256: string;
        };
        /** AcousticInputs */
        AcousticInputs: {
            audio: components["schemas"]["AcousticAssetRef"];
            /** @default null */
            lip: components["schemas"]["AcousticAssetRef"] | null;
            /** @default null */
            textgrid: components["schemas"]["AcousticAssetRef"] | null;
        };
        /** AcousticMetadata */
        AcousticMetadata: {
            /**
             * Adapter Version
             * @default m01-adapter/1
             * @constant
             */
            adapter_version: "m01-adapter/1";
            /**
             * Algorithm Id
             * @default m01.parameter_estimation
             * @constant
             */
            algorithm_id: "m01.parameter_estimation";
            /**
             * Algorithm Version
             * @default legacy-numeric/1
             * @constant
             */
            algorithm_version: "legacy-numeric/1";
            /** Backends */
            backends: components["schemas"]["AcousticBackendObservation"][];
            config: components["schemas"]["AcousticConfigSnapshot"];
            /** Config Sha256 */
            config_sha256: string;
            /** Core Version */
            core_version: string;
            decoded: components["schemas"]["AcousticDecodedAudio"];
            /** Inputs */
            inputs: components["schemas"]["AcousticInputSnapshot"][];
            /** Project Id */
            project_id: string;
            /**
             * Schema Version
             * @default m01/1
             * @constant
             */
            schema_version: "m01/1";
            /** Source Ids */
            source_ids: string[];
        };
        /** AcousticNumericColumn */
        AcousticNumericColumn: {
            /**
             * Key
             * @enum {string}
             */
            key: "pF0" | "rF0" | "pF1" | "pF2" | "pF3" | "pF4" | "pB1" | "pB2" | "pB3" | "pB4" | "H1_pF0" | "H1_rF0" | "H2_pF0" | "H2_rF0" | "H4_pF0" | "H4_rF0" | "A1_pF0" | "A1_rF0" | "A2_pF0" | "A2_rF0" | "A3_pF0" | "A3_rF0" | "H1H2u_pF0" | "H1H2u_rF0" | "H2H4u_pF0" | "H2H4u_rF0" | "H1A1u_pF0" | "H1A1u_rF0" | "H1A2u_pF0" | "H1A2u_rF0" | "H1A3u_pF0" | "H1A3u_rF0" | "H1A1c_pF0" | "H1A1c_rF0" | "H1A2c_pF0" | "H1A2c_rF0" | "H1A3c_pF0" | "H1A3c_rF0" | "H1H2c_pF0" | "H1H2c_rF0" | "H2H4c_pF0" | "H2H4c_rF0" | "H2K_pF0" | "H2K_rF0" | "H5K_pF0" | "H5K_rF0" | "H42Ku_pF0" | "H42Ku_rF0" | "H2KH5Ku_pF0" | "H2KH5Ku_rF0" | "H42Kc_pF0" | "H42Kc_rF0" | "H2KH5Kc_pF0" | "H2KH5Kc_rF0" | "CPP_pF0" | "CPP_rF0" | "Intensity" | "HNR05_pF0" | "HNR15_pF0" | "HNR25_pF0" | "HNR35_pF0" | "HNR05_rF0" | "HNR15_rF0" | "HNR25_rF0" | "HNR35_rF0" | "SHR_pF0" | "SHR_rF0" | "SpectralSlope_pF0" | "SpectralSlope_rF0" | "Jitter_Local" | "Jitter_RAP" | "Jitter_PPQ5" | "Shimmer_Local" | "Shimmer_APQ3" | "Shimmer_APQ5" | "Shimmer_APQ11" | "LipArea" | "LipWidth" | "LipOpen" | "LipCirc" | "SOE_pF0" | "SOE_rF0";
            /** Label */
            label: string;
            /**
             * Nonfinite
             * @description 0 finite; 1 NaN; 2 +Infinity; 3 -Infinity
             */
            nonfinite: (0 | 1 | 2 | 3)[];
            /**
             * Reason
             * @description Unknown cause is explicit. No inferred unvoiced/failed label from a legacy NaN.
             */
            reason: ("legacy_nonfinite_unknown" | null)[];
            /**
             * Scope
             * @enum {string}
             */
            scope: "catalog" | "legacy_service_extension";
            /** Unit */
            unit: string;
            /** Values */
            values: (number | null)[];
        };
        /** AcousticRequest */
        AcousticRequest: {
            config?: components["schemas"]["AcousticConfigSnapshot"];
            /** Idempotency Key */
            idempotency_key: string;
            inputs: components["schemas"]["AcousticInputs"];
            /**
             * Operation
             * @default acoustic_analysis
             * @constant
             */
            operation: "acoustic_analysis";
            /** Project Id */
            project_id: string;
            /**
             * Schema Version
             * @default m01/1
             * @constant
             */
            schema_version: "m01/1";
        };
        /** AcousticResult */
        AcousticResult: {
            /** Column Order */
            column_order: string[];
            metadata: components["schemas"]["AcousticMetadata"];
            /** Numeric */
            numeric: components["schemas"]["AcousticNumericColumn"][];
            /**
             * Schema Version
             * @default m01/1
             * @constant
             */
            schema_version: "m01/1";
            /** Text */
            text: components["schemas"]["AcousticTextColumn"][];
            /** Times S */
            times_s: number[];
        };
        /** AcousticResultFile */
        AcousticResultFile: {
            /** Asset Id */
            asset_id: string;
            /** Expires At */
            expires_at: number | null;
            /**
             * Format
             * @enum {string}
             */
            format: "xlsx" | "sqlite";
            /** Sha256 */
            sha256: string;
            /** Size Bytes */
            size_bytes: number;
        };
        /** AcousticSelection */
        AcousticSelection: {
            /** Keys */
            keys?: ("pF0" | "rF0" | "pF1" | "pF2" | "pF3" | "pF4" | "pB1" | "pB2" | "pB3" | "pB4" | "H1_pF0" | "H1_rF0" | "H2_pF0" | "H2_rF0" | "H4_pF0" | "H4_rF0" | "A1_pF0" | "A1_rF0" | "A2_pF0" | "A2_rF0" | "A3_pF0" | "A3_rF0" | "H1H2u_pF0" | "H1H2u_rF0" | "H2H4u_pF0" | "H2H4u_rF0" | "H1A1u_pF0" | "H1A1u_rF0" | "H1A2u_pF0" | "H1A2u_rF0" | "H1A3u_pF0" | "H1A3u_rF0" | "H1A1c_pF0" | "H1A1c_rF0" | "H1A2c_pF0" | "H1A2c_rF0" | "H1A3c_pF0" | "H1A3c_rF0" | "H1H2c_pF0" | "H1H2c_rF0" | "H2H4c_pF0" | "H2H4c_rF0" | "H2K_pF0" | "H2K_rF0" | "H5K_pF0" | "H5K_rF0" | "H42Ku_pF0" | "H42Ku_rF0" | "H2KH5Ku_pF0" | "H2KH5Ku_rF0" | "H42Kc_pF0" | "H42Kc_rF0" | "H2KH5Kc_pF0" | "H2KH5Kc_rF0" | "CPP_pF0" | "CPP_rF0" | "Intensity" | "HNR05_pF0" | "HNR15_pF0" | "HNR25_pF0" | "HNR35_pF0" | "HNR05_rF0" | "HNR15_rF0" | "HNR25_rF0" | "HNR35_rF0" | "SHR_pF0" | "SHR_rF0" | "SpectralSlope_pF0" | "SpectralSlope_rF0" | "Jitter_Local" | "Jitter_RAP" | "Jitter_PPQ5" | "Shimmer_Local" | "Shimmer_APQ3" | "Shimmer_APQ5" | "Shimmer_APQ11" | "LipArea" | "LipWidth" | "LipOpen" | "LipCirc")[];
            /**
             * Mode
             * @default catalog
             * @enum {string}
             */
            mode: "catalog" | "legacy_service";
        };
        /** AcousticSettings */
        AcousticSettings: {
            /**
             * Energy Window Ms
             * @default 40
             */
            energy_window_ms: number;
            /**
             * Frameshift Ms
             * @default 5
             */
            frameshift_ms: number;
            /**
             * Lip Smooth Win Size
             * @default 0
             */
            lip_smooth_win_size: number;
            /**
             * Max F0
             * @default 880
             */
            max_f0: number;
            /**
             * Max Formant
             * @default 6000
             */
            max_formant: number;
            /**
             * Min F0
             * @default 60
             */
            min_f0: number;
            /**
             * N Periods
             * @default 3
             */
            n_periods: number;
            /**
             * Num Formants
             * @default 5
             */
            num_formants: number;
            /**
             * Only Voiced
             * @default true
             */
            only_voiced: boolean;
            /**
             * Reaper Hilbert
             * @default true
             */
            reaper_hilbert: boolean;
            /**
             * Reaper No Highpass
             * @default false
             */
            reaper_no_highpass: boolean;
            /**
             * Silence Threshold
             * @default 0.03
             */
            silence_threshold: number;
            /**
             * Smooth Win Size
             * @default 10
             */
            smooth_win_size: number;
            /**
             * Windowsize Ms
             * @default 40
             */
            windowsize_ms: number;
        };
        /** AcousticTextColumn */
        AcousticTextColumn: {
            /** Key */
            key: string;
            /** Values */
            values: string[];
        };
        /** AssetList */
        AssetList: {
            /** Assets */
            assets: components["schemas"]["AssetView"][];
        };
        /** AssetView */
        AssetView: {
            /** Created At */
            created_at: number;
            /** Error Code */
            error_code: string | null;
            /** Expected Bytes */
            expected_bytes: number | null;
            /** Expires At */
            expires_at: number;
            /** Id */
            id: string;
            /**
             * Kind
             * @enum {string}
             */
            kind: "input" | "result" | "archive" | "temporary";
            /** Name */
            name: string;
            /** Project Id */
            project_id: string;
            /** Reserved Bytes */
            reserved_bytes: number;
            /** Sha256 */
            sha256: string | null;
            /** Size Bytes */
            size_bytes: number;
            /**
             * State
             * @enum {string}
             */
            state: "uploading" | "ready" | "deleting" | "delete_failed" | "deleted";
        };
        /** Audio */
        Audio: {
            /** Channel Roles */
            channel_roles: string[];
            /** Channels */
            channels: number;
            /**
             * Origin
             * @enum {string}
             */
            origin: "file" | "recording" | "generated";
            /** Sample Count */
            sample_count: number;
            /** Sample Rate Hz */
            sample_rate_hz: number;
        };
        /** Capabilities */
        Capabilities: {
            /** Algorithms */
            algorithms: string[];
            /**
             * Api Version
             * @default 1.1.0
             */
            api_version: string;
            /** Limitations */
            limitations: string[];
            /**
             * Stage
             * @default P02
             * @enum {string}
             */
            stage: "P02" | "P05" | "P06" | "P07";
            /**
             * Storage Operations
             * @default []
             */
            storage_operations: string[];
            /**
             * Task Operations
             * @default []
             */
            task_operations: string[];
        };
        /** Challenge */
        Challenge: {
            /** Csrf Token */
            csrf_token: string;
        };
        /** DeleteImpact */
        DeleteImpact: {
            /** Active Jobs */
            active_jobs: string[];
        };
        /** FileConfig */
        FileConfig: {
            /** Inputs */
            inputs?: string[];
            /**
             * Max Output Bytes
             * @default 16777216
             */
            max_output_bytes: number;
            /**
             * Probe Bytes
             * @default 16384
             */
            probe_bytes: number;
            /**
             * Probe Files
             * @default 2
             */
            probe_files: number;
        };
        /** FileJobInput */
        FileJobInput: {
            config?: components["schemas"]["FileConfig"];
            /** Idempotency Key */
            idempotency_key: string;
            /**
             * Operation
             * @enum {string}
             */
            operation: "storage_check" | "archive_zip" | "extract_zip";
            /** Project Id */
            project_id: string;
        };
        /** FileManifest */
        FileManifest: {
            /**
             * Complete
             * @default true
             * @constant
             */
            complete: true;
            /** Core Version */
            core_version: string;
            /** Files */
            files: components["schemas"]["ResultFile"][];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "managed_files";
        };
        /** FinalizeInput */
        FinalizeInput: {
            /** Sha256 */
            sha256?: string | null;
        };
        /** HTTPValidationError */
        HTTPValidationError: {
            /** Detail */
            detail?: components["schemas"]["ValidationError"][];
        };
        /** Health */
        Health: {
            /**
             * Api Version
             * @default 1.1.0
             */
            api_version: string;
            /** App Version */
            app_version: string;
            /** Core Version */
            core_version: string;
            /**
             * Mode
             * @enum {string}
             */
            mode: "local" | "server";
            /**
             * Status
             * @default ok
             * @constant
             */
            status: "ok";
        };
        /** JobEvent */
        JobEvent: {
            /** Code */
            code: string;
            /** Created At */
            created_at: number;
            /** Progress */
            progress: number;
            /** Sequence */
            sequence: number;
            /**
             * State
             * @enum {string}
             */
            state: "queued" | "running" | "cancel_requested" | "cancelled" | "failed" | "interrupted" | "succeeded";
        };
        /** JobEvents */
        JobEvents: {
            /** Events */
            events: components["schemas"]["JobEvent"][];
        };
        /** JobInput */
        JobInput: {
            config?: components["schemas"]["ProbeConfig"];
            /** Idempotency Key */
            idempotency_key: string;
            /**
             * Operation
             * @default pipeline_check
             * @constant
             */
            operation: "pipeline_check";
            /** Project Id */
            project_id: string;
        };
        /** JobList */
        JobList: {
            /** Jobs */
            jobs: components["schemas"]["JobView"][];
        };
        /** JobManifest */
        JobManifest: {
            /**
             * Complete
             * @default true
             * @constant
             */
            complete: true;
            /** Core Version */
            core_version: string;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "pipeline_check_metadata";
            /** Sample Count */
            sample_count: number;
            /** Sha256 */
            sha256: string;
        };
        /** JobView */
        JobView: {
            /** Created At */
            created_at: number;
            /** Error Code */
            error_code: string | null;
            /** Generation */
            generation: number;
            /** Id */
            id: string;
            /**
             * Operation
             * @default pipeline_check
             * @enum {string}
             */
            operation: "pipeline_check" | "storage_check" | "archive_zip" | "extract_zip";
            /** Progress */
            progress: number;
            /** Project Id */
            project_id: string;
            /** Result Manifest */
            result_manifest: components["schemas"]["JobManifest"] | components["schemas"]["FileManifest"] | null;
            /** Retry Of */
            retry_of?: string | null;
            /**
             * State
             * @enum {string}
             */
            state: "queued" | "running" | "cancel_requested" | "cancelled" | "failed" | "interrupted" | "succeeded";
            /** Updated At */
            updated_at: number;
        };
        /** LoginInput */
        LoginInput: {
            /** Password */
            password: string;
            /** Username */
            username: string;
        };
        /** ProbeConfig */
        ProbeConfig: {
            /**
             * Sample Count
             * @default 4096
             */
            sample_count: number;
            /**
             * Seed
             * @default 0
             */
            seed: number;
        };
        /** ProjectInput */
        ProjectInput: {
            /** Name */
            name: string;
        };
        /** ProjectList */
        ProjectList: {
            /** Projects */
            projects: components["schemas"]["ProjectView"][];
        };
        /** ProjectView */
        ProjectView: {
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Id */
            id: string;
            /** Name */
            name: string;
        };
        /** ResultFile */
        ResultFile: {
            /** Expires At */
            expires_at: number;
            /** Id */
            id: string;
            /**
             * Kind
             * @enum {string}
             */
            kind: "result" | "archive";
            /** Name */
            name: string;
            /** Sha256 */
            sha256: string;
            /** Size Bytes */
            size_bytes: number;
        };
        /** ResultManifestEnvelope */
        ResultManifestEnvelope: {
            /** Manifest */
            manifest: components["schemas"]["JobManifest"] | components["schemas"]["FileManifest"] | components["schemas"]["AcousticFileManifest"];
        };
        /** RetryInput */
        RetryInput: {
            /** Idempotency Key */
            idempotency_key: string;
        };
        /**
         * Selection
         * @description Integer sample-frame interval [start_sample, end_sample).
         */
        Selection: {
            /** End Sample */
            end_sample: number;
            /** Sample Rate Hz */
            sample_rate_hz: number;
            /** Start Sample */
            start_sample: number;
        };
        /** SessionView */
        SessionView: {
            /** Csrf Token */
            csrf_token: string;
            /**
             * Expires At
             * Format: date-time
             */
            expires_at: string;
            user: components["schemas"]["UserView"];
        };
        /** StorageUsage */
        StorageUsage: {
            /** Available Bytes */
            available_bytes: number;
            /** Frozen */
            frozen: boolean;
            /** Quota Bytes */
            quota_bytes: number;
            /** Ready */
            ready: boolean;
            /** Reserved Bytes */
            reserved_bytes: number;
            /** Used Bytes */
            used_bytes: number;
        };
        /** Track */
        Track: {
            /** Analysis Config Hash */
            analysis_config_hash: string;
            /** Backend */
            backend: string;
            /** Parameter Key */
            parameter_key: string;
            /** Reason */
            reason: (string | null)[];
            /** Source Ids */
            source_ids: string[];
            /** Times S */
            times_s: number[];
            /**
             * Unit
             * @description Explicit scientific unit, e.g. Hz, dB, s, %, or 1; parameter mapping is audited in P03.
             */
            unit: string;
            /** Validity */
            validity: ("valid" | "unvoiced" | "missing" | "failed")[];
            /** Values */
            values: (number | null)[];
        };
        /** UploadInput */
        UploadInput: {
            /** Expected Bytes */
            expected_bytes?: number | null;
            /** Idempotency Key */
            idempotency_key: string;
            /** Name */
            name: string;
            /**
             * Project Id
             * Format: uuid
             */
            project_id: string;
        };
        /** UserView */
        UserView: {
            /** Id */
            id: string;
            /** Username */
            username: string;
        };
        /** ValidationError */
        ValidationError: {
            /** Context */
            ctx?: Record<string, never>;
            /** Input */
            input?: unknown;
            /** Location */
            loc: (string | number)[];
            /** Message */
            msg: string;
            /** Error Type */
            type: string;
        };
        /** Viewport */
        Viewport: {
            audio: components["schemas"]["Audio"];
            selection: components["schemas"]["Selection"];
            /** Tracks */
            tracks: components["schemas"]["Track"][];
        };
    };
    responses: never;
    parameters: never;
    requestBodies: never;
    headers: never;
    pathItems: never;
}
export type $defs = Record<string, never>;
export interface operations {
    list_assets: {
        parameters: {
            query: {
                project_id: string;
                order?: "expires" | "size" | "created";
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AssetList"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_asset: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                asset_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AssetView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    delete_asset: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                asset_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AssetView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    download_asset: {
        parameters: {
            query?: {
                expected_account?: string | null;
            };
            header?: never;
            path: {
                asset_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/octet-stream": unknown;
                };
            };
            /** @description Partial content */
            206: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Invalid or unsatisfiable single range */
            416: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_delete_impact: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                asset_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["DeleteImpact"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_login_challenge: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Challenge"];
                };
            };
        };
    };
    login_account: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["LoginInput"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SessionView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    logout_account: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
        };
    };
    get_current_account: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SessionView"];
                };
            };
        };
    };
    get_capabilities: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Capabilities"];
                };
            };
        };
    };
    get_health: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Health"];
                };
            };
        };
    };
    list_jobs: {
        parameters: {
            query: {
                project_id: string;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["JobList"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    create_job: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["JobInput"] | components["schemas"]["FileJobInput"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["JobView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_job: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                job_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["JobView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    cancel_job: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                job_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["JobView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_job_events: {
        parameters: {
            query?: {
                after?: number;
            };
            header?: never;
            path: {
                job_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["JobEvents"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    retry_job: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                job_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["RetryInput"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["JobView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_projects: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProjectList"];
                };
            };
        };
    };
    create_project: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["ProjectInput"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProjectView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_project: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                project_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProjectView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    rename_project: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                project_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["ProjectInput"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProjectView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_storage_usage: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["StorageUsage"];
                };
            };
        };
    };
    create_upload: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["UploadInput"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AssetView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    append_upload_block: {
        parameters: {
            query: {
                offset: number;
            };
            header?: never;
            path: {
                asset_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/octet-stream": string;
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AssetView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    finalize_upload: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                asset_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["FinalizeInput"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AssetView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
}
