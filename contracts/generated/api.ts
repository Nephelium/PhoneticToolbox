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
    "/api/v1/assets/{asset_id}/parameters": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Parameters */
        get: operations["asset_parameter_table"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/assets/{asset_id}/spectrogram": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Spectrogram */
        get: operations["asset_spectrogram_preview"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/assets/{asset_id}/textgrid": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Textgrid */
        get: operations["preview_textgrid"];
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
    "/api/v1/jobs/batches/create": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Create Batch */
        post: operations["create_acoustic_batch"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/batches/list": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** List Batches */
        get: operations["list_acoustic_batches"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/batches/{batch_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Batch */
        get: operations["get_acoustic_batch"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/batches/{batch_id}/cancel": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Cancel Batch */
        post: operations["cancel_acoustic_batch"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/egg/create": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Egg */
        post: operations["create_egg_job"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/egg/fonts": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Egg Fonts */
        post: operations["check_egg_export_fonts"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/local-inputs": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Local Input */
        post: operations["register_local_acoustic_input"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/local-lip-conversion": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Convert Lip */
        post: operations["convert_local_legacy_lip"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/local-results/{asset_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Local Result */
        get: operations["read_local_acoustic_result"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/lpc/create": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Lpc */
        post: operations["create_lpc_job"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/lpc/fonts": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Lpc Fonts */
        post: operations["check_lpc_export_fonts"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/m05/catalog": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** M05 Catalog */
        get: operations["lip_catalog"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/m05/create": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** M05 Create */
        post: operations["create_lip_job"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/m05/uploads": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** M05 Begin */
        post: operations["begin_local_lip_video"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/m05/uploads/{key}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        /** M05 Block */
        put: operations["write_local_lip_video"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/m05/uploads/{key}/abort": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** M05 Abort */
        post: operations["abort_local_lip_video"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/m05/uploads/{key}/finalize": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** M05 Finish */
        post: operations["finalize_local_lip_video"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/m05/{key}/repeat": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** M05 Repeat */
        post: operations["render_lip_animation"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/m06/create": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Create M06 */
        post: operations["create_speech_synthesis_job"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/m07/create": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Create M07 */
        post: operations["create_phonation_synthesis_job"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/m08/create": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** M08 */
        post: operations["create_m08_job"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/m08/history": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** M08 History */
        post: operations["m08_history"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/m08/list/{project_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** M08 List */
        get: operations["list_m08_results"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/m08/remove": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** M08 Remove */
        post: operations["remove_m08_results"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/m08/rename": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** M08 Rename */
        post: operations["rename_m08_results"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/m08/save": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** M08 Save */
        post: operations["save_m08_result"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/m11/catalog": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** M11 Catalog */
        get: operations["mfa_component_catalog"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/m11/component": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Component */
        post: operations["manage_local_mfa_component"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/m11/create": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** M11 Create */
        post: operations["create_mfa_alignment_job"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/m14/create": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Create M14 */
        post: operations["create_phonology_job"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/parents/latest": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Parent */
        get: operations["find_acoustic_parent"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/jobs/spec2wav/create": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Reconstruct */
        post: operations["create_spec2wav_job"];
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
    "/api/v1/preview/parameters": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Parameters */
        post: operations["local_parameter_table"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/preview/spectrogram": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Spectrogram */
        post: operations["local_spectrogram_preview"];
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
             * Policy Version
             * @default 1
             * @enum {integer}
             */
            policy_version: 1 | 2;
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
        /** AcousticManagedFile */
        AcousticManagedFile: {
            /** Expires At */
            expires_at: number | null;
            /** Id */
            id: string;
            /**
             * Kind
             * @default result
             * @constant
             */
            kind: "result";
            /** Name */
            name: string;
            /** Sha256 */
            sha256: string;
            /** Size Bytes */
            size_bytes: number;
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
        /** AcousticTaskManifest */
        AcousticTaskManifest: {
            /**
             * Complete
             * @default true
             * @constant
             */
            complete: true;
            /** Core Version */
            core_version: string;
            /** Files */
            files: components["schemas"]["AcousticManagedFile"][];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "managed_acoustic_files";
            /**
             * Operation
             * @enum {string}
             */
            operation: "acoustic_analysis" | "textgrid_segment";
            /**
             * Policy Version
             * @default 1
             * @enum {integer}
             */
            policy_version: 1 | 2;
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
            /**
             * Policy Version
             * @default 1
             * @enum {integer}
             */
            policy_version: 1 | 2;
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
        /** BatchInputs */
        BatchInputs: {
            audio: components["schemas"]["AcousticAssetRef"];
            legacy_result?: components["schemas"]["AcousticAssetRef"] | null;
            lip?: components["schemas"]["AcousticAssetRef"] | null;
            parent_result?: components["schemas"]["AcousticAssetRef"] | null;
            textgrid?: components["schemas"]["AcousticAssetRef"] | null;
        };
        /** BatchList */
        BatchList: {
            /** Batches */
            batches: components["schemas"]["BatchView"][];
        };
        /** BatchRequest */
        BatchRequest: {
            config?: components["schemas"]["AcousticConfigSnapshot"] | null;
            /** Idempotency Key */
            idempotency_key: string;
            /** Inputs */
            inputs: components["schemas"]["BatchInputs"][];
            /** Layer */
            layer?: string | null;
            /**
             * Operation
             * @enum {string}
             */
            operation: "acoustic_analysis" | "textgrid_segment";
            /** Project Id */
            project_id: string;
            /**
             * Schema Version
             * @default m01-batch/1
             * @constant
             */
            schema_version: "m01-batch/1";
        };
        /** BatchView */
        BatchView: {
            /** Audio Names */
            audio_names: string[];
            /** Cancel Requested */
            cancel_requested: boolean;
            /** Created At */
            created_at: number;
            /** Id */
            id: string;
            /**
             * Operation
             * @enum {string}
             */
            operation: "acoustic_analysis" | "textgrid_segment";
            /** Project Id */
            project_id: string;
            /** Request Sha256 */
            request_sha256: string;
            summary: components["schemas"]["AcousticBatchSummary"];
            /** Updated At */
            updated_at: number;
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
        /** EggInverseData */
        EggInverseData: {
            /** Audio Db */
            audio_db: number[];
            /** Audio Values */
            audio_values: number[];
            /** Egg Db */
            egg_db: number[];
            /** Egg Values */
            egg_values: number[];
            /** Frequencies Hz */
            frequencies_hz: number[];
            /** Inverse Db */
            inverse_db: number[];
            /** Inverse Values */
            inverse_values: number[];
            /** Relative Times S */
            relative_times_s: number[];
            /**
             * Schema Version
             * @default egg-inverse-view/1
             * @constant
             */
            schema_version: "egg-inverse-view/1";
            /**
             * Spectral Policy
             * @default pad-44100-periodic-hamming-fft-80db-floor
             * @constant
             */
            spectral_policy: "pad-44100-periodic-hamming-fft-80db-floor";
        };
        /** EggManifest */
        EggManifest: {
            /**
             * Complete
             * @default true
             * @constant
             */
            complete: true;
            /** Core Version */
            core_version: string;
            /** Files */
            files: components["schemas"]["AcousticManagedFile"][];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "managed_egg_files";
            /**
             * Operation
             * @default egg_analysis
             * @constant
             */
            operation: "egg_analysis";
            /**
             * Policy Version
             * @default 1
             * @enum {integer}
             */
            policy_version: 1 | 2;
        };
        /** EggPreviewData */
        EggPreviewData: {
            audio: components["schemas"]["EggSeries"];
            cq: components["schemas"]["EggSeries"];
            /**
             * Display Policy
             * @default legacy-roi-extent; grayscale-raster-only
             * @constant
             */
            display_policy: "legacy-roi-extent; grayscale-raster-only";
            egg: components["schemas"]["EggSeries"];
            /** Gci */
            gci: number[];
            gci_f0: components["schemas"]["EggSeries"];
            /** Goi */
            goi: number[];
            /** Micro Center */
            micro_center: number;
            /**
             * Micro Event Policy
             * @default raw-50ms-padding
             * @constant
             */
            micro_event_policy: "raw-50ms-padding";
            /**
             * Micro Sample Stride
             * @default 1
             */
            micro_sample_stride: number;
            /**
             * Micro Wave Policy
             * @default raw-100ms-padding-filter-crop
             * @constant
             */
            micro_wave_policy: "raw-100ms-padding-filter-crop";
            /** Micro Width Ms */
            micro_width_ms: number;
            /** Movement */
            movement: [
                number,
                string
            ][];
            praat: components["schemas"]["EggSeries"];
            /** Raster Shape */
            raster_shape: [
                number,
                number
            ];
            /**
             * Schema Version
             * @default egg-preview/1
             * @constant
             */
            schema_version: "egg-preview/1";
            /** Spectral Extent */
            spectral_extent: [
                number,
                number,
                number,
                number
            ];
            /** Spectral Shape */
            spectral_shape: [
                number,
                number
            ];
            sq: components["schemas"]["EggSeries"];
        };
        /** EggRequest */
        EggRequest: {
            audio: components["schemas"]["AcousticAssetRef"];
            config: components["schemas"]["EggTaskConfig"];
            /** Idempotency Key */
            idempotency_key: string;
            /** Project Id */
            project_id: string;
            /**
             * Schema Version
             * @default m03/1
             * @constant
             */
            schema_version: "m03/1";
        };
        /** EggSeries */
        EggSeries: {
            /** Times */
            times: number[];
            /** Values */
            values: (number | null)[];
        };
        /** EggTaskConfig */
        EggTaskConfig: {
            /**
             * Auto Prominence
             * @default true
             */
            auto_prominence: boolean;
            /**
             * Export Policy
             * @default sample-aligned/1
             * @constant
             */
            export_policy: "sample-aligned/1";
            /**
             * Flip Channels
             * @default false
             */
            flip_channels: boolean;
            font?: components["schemas"]["FigureFontSnapshot"] | null;
            /**
             * Gci Method
             * @default slope
             * @enum {string}
             */
            gci_method: "slope" | "scale";
            /**
             * Generate Images
             * @default false
             */
            generate_images: boolean;
            /**
             * Glottal Movement
             * @default false
             */
            glottal_movement: boolean;
            /**
             * Goi Method
             * @default scale
             * @enum {string}
             */
            goi_method: "slope" | "scale";
            /**
             * Highpass Cutoff
             * @default 25
             */
            highpass_cutoff: number;
            /**
             * Keep Gci F0
             * @default true
             */
            keep_gci_f0: boolean;
            /**
             * Keep Praat F0
             * @default true
             */
            keep_praat_f0: boolean;
            /**
             * Lowpass Cutoff
             * @default 1000
             */
            lowpass_cutoff: number;
            /** Lp Order */
            lp_order?: number | null;
            /** Micro Center */
            micro_center?: number | null;
            /**
             * Micro Width Ms
             * @default 50
             */
            micro_width_ms: number;
            /**
             * Mode
             * @default single
             * @enum {string}
             */
            mode: "single" | "batch" | "inverse" | "preview";
            /**
             * Peak Prominence
             * @default 0.01
             */
            peak_prominence: number;
            /** Roi End */
            roi_end?: number | null;
            /**
             * Roi Start
             * @default 0
             */
            roi_start: number;
            /**
             * Signal Mode
             * @default filtered
             * @enum {string}
             */
            signal_mode: "raw" | "filtered";
            /**
             * Silence Threshold
             * @default 0.01
             */
            silence_threshold: number;
            /**
             * Spec Vmax
             * @default -10
             */
            spec_vmax: number;
            /**
             * Spec Vmin
             * @default -70
             */
            spec_vmin: number;
            /**
             * Spec Window Ms
             * @default 20
             */
            spec_window_ms: number;
            /**
             * Valley Prominence
             * @default 0.01
             */
            valley_prominence: number;
        };
        /** FigureFontSnapshot */
        FigureFontSnapshot: {
            /**
             * Ipa
             * @default Doulos SIL
             * @constant
             */
            ipa: "Doulos SIL";
            /**
             * Latin
             * @default Segoe UI
             */
            latin: string;
            /**
             * Schema Version
             * @default font/1
             * @constant
             */
            schema_version: "font/1";
            /**
             * Size Px
             * @default 12
             */
            size_px: number;
            /**
             * Zh
             * @default Microsoft YaHei
             */
            zh: string;
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
            /**
             * Policy Version
             * @default 1
             * @enum {integer}
             */
            policy_version: 1 | 2;
        };
        /** FinalizeInput */
        FinalizeInput: {
            /** Sha256 */
            sha256?: string | null;
        };
        /** FontCheckItem */
        FontCheckItem: {
            /** Available */
            available: boolean;
            /** Family */
            family?: string | null;
            /** Requested */
            requested: string;
            /**
             * Role
             * @enum {string}
             */
            role: "zh" | "latin" | "ipa";
            /** Sha256 */
            sha256?: string | null;
        };
        /** FontPreflight */
        FontPreflight: {
            /** Available */
            available: boolean;
            /** Fonts */
            fonts: components["schemas"]["FontCheckItem"][];
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
        /** ImagePoint */
        ImagePoint: {
            /** X */
            x: number;
            /** Y */
            y: number;
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
            operation: "pipeline_check" | "storage_check" | "archive_zip" | "extract_zip" | "acoustic_analysis" | "textgrid_segment" | "spectrogram_to_audio" | "egg_analysis" | "lpc_analysis" | "pitch_manipulation" | "phonology_induction" | "speech_synthesis" | "phonation_synthesis" | "mfa_alignment" | "lip_analysis";
            /** Progress */
            progress: number;
            /** Project Id */
            project_id: string;
            /** Result Manifest */
            result_manifest: components["schemas"]["JobManifest"] | components["schemas"]["FileManifest"] | components["schemas"]["AcousticTaskManifest"] | components["schemas"]["Spec2WavManifest"] | components["schemas"]["EggManifest"] | components["schemas"]["LpcManifest"] | components["schemas"]["M08Manifest"] | components["schemas"]["M14Manifest"] | components["schemas"]["M06Manifest"] | components["schemas"]["M07Manifest"] | components["schemas"]["M11Manifest"] | components["schemas"]["M05Manifest"] | null;
            /** Retry Of */
            retry_of?: string | null;
            /**
             * State
             * @enum {string}
             */
            state: "queued" | "running" | "cancel_requested" | "cancelled" | "failed" | "interrupted" | "succeeded";
            /** Updated At */
            updated_at: number;
            /** Waiting Reason */
            waiting_reason?: string | null;
        };
        /** LocalM11ComponentRequest */
        LocalM11ComponentRequest: {
            /**
             * Action
             * @enum {string}
             */
            action: "check" | "import";
            /** Archive */
            archive?: string | null;
            /** Dictionary */
            dictionary: string;
            /** Manifest */
            manifest?: string | null;
            /** Model */
            model: string;
            /** Runtime */
            runtime?: string | null;
            /** Trusted Manifest Sha256 */
            trusted_manifest_sha256?: string | null;
        };
        /** LoginInput */
        LoginInput: {
            /** Password */
            password: string;
            /** Username */
            username: string;
        };
        /** LpcManifest */
        LpcManifest: {
            /**
             * Complete
             * @default true
             * @constant
             */
            complete: true;
            /** Core Version */
            core_version: string;
            /** Files */
            files: components["schemas"]["AcousticManagedFile"][];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "managed_lpc_files";
            /**
             * Operation
             * @default lpc_analysis
             * @constant
             */
            operation: "lpc_analysis";
            /**
             * Policy Version
             * @default 1
             * @enum {integer}
             */
            policy_version: 1 | 2;
        };
        /** LpcRequest */
        LpcRequest: {
            audio: components["schemas"]["AcousticAssetRef"];
            config: components["schemas"]["LpcTaskConfig"];
            /** Idempotency Key */
            idempotency_key: string;
            /** Project Id */
            project_id: string;
            /**
             * Schema Version
             * @default m04/1
             * @constant
             */
            schema_version: "m04/1";
            textgrid?: components["schemas"]["AcousticAssetRef"] | null;
        };
        /** LpcSpectrumData */
        LpcSpectrumData: {
            /** Amp Max Db */
            amp_max_db: number;
            /** Amp Min Db */
            amp_min_db: number;
            /** Frequencies Hz */
            frequencies_hz: number[];
            /** Magnitude Db */
            magnitude_db: number[];
        };
        /** LpcTaskConfig */
        LpcTaskConfig: {
            /**
             * Amp Max Db
             * @default 35
             */
            amp_max_db: number;
            /**
             * Amp Min Db
             * @default -5
             */
            amp_min_db: number;
            /**
             * Dynamic Y
             * @default false
             */
            dynamic_y: boolean;
            font?: components["schemas"]["FigureFontSnapshot"];
            /**
             * Freq Max Hz
             * @default 8000
             */
            freq_max_hz: number;
            /**
             * Order
             * @default 50
             */
            order: number;
            /** Roi End */
            roi_end: number;
            /**
             * Roi Start
             * @default 0
             */
            roi_start: number;
            /** Tier Name */
            tier_name?: string | null;
        };
        /** M05Block */
        M05Block: {
            /** Base64 */
            base64: string;
            /** Offset */
            offset: number;
        };
        /** M05Config */
        M05Config: {
            /**
             * Animation
             * @default none
             * @enum {string}
             */
            animation: "none" | "mp4" | "gif";
            /**
             * Cutoff Hz
             * @default 15
             */
            cutoff_hz: number;
            /**
             * Filter Enabled
             * @default true
             */
            filter_enabled: boolean;
            /**
             * Offset
             * @default 0
             */
            offset: number;
            /**
             * Quality
             * @default standard
             * @enum {string}
             */
            quality: "high" | "standard" | "small";
        };
        /** M05Manifest */
        M05Manifest: {
            /**
             * Complete
             * @default true
             * @constant
             */
            complete: true;
            /** Core Version */
            core_version: string;
            /** Files */
            files: components["schemas"]["AcousticManagedFile"][];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "managed_m05_files";
            /**
             * Operation
             * @default lip_analysis
             * @constant
             */
            operation: "lip_analysis";
            /**
             * Policy Version
             * @default 1
             * @enum {integer}
             */
            policy_version: 1 | 2;
        };
        /** M05Request */
        M05Request: {
            config?: components["schemas"]["M05Config"];
            /** Idempotency Key */
            idempotency_key: string;
            /** Project Id */
            project_id: string;
            /**
             * Schema Version
             * @default m05/1
             * @constant
             */
            schema_version: "m05/1";
            video: components["schemas"]["AcousticAssetRef"];
        };
        /** M05Upload */
        M05Upload: {
            /** Name */
            name: string;
            /** Size */
            size: number;
        };
        /** M06Manifest */
        M06Manifest: {
            /**
             * Complete
             * @default true
             * @constant
             */
            complete: true;
            /** Core Version */
            core_version: string;
            /** Files */
            files: components["schemas"]["AcousticManagedFile"][];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "managed_m06_files";
            /**
             * Operation
             * @default speech_synthesis
             * @constant
             */
            operation: "speech_synthesis";
            /**
             * Policy Version
             * @default 1
             * @enum {integer}
             */
            policy_version: 1 | 2;
        };
        /** M06Request */
        M06Request: {
            /**
             * Action
             * @enum {string}
             */
            action: "generate" | "synthesize" | "extract";
            audio?: components["schemas"]["AcousticAssetRef"] | null;
            /** Idempotency Key */
            idempotency_key: string;
            parameters: components["schemas"]["AcousticAssetRef"];
            /** Project Id */
            project_id: string;
            /**
             * Schema Version
             * @default m06/1
             * @constant
             */
            schema_version: "m06/1";
        };
        /** M07Analysis */
        M07Analysis: {
            /**
             * F0 Backend
             * @default parselmouth
             * @enum {string}
             */
            f0_backend: "parselmouth" | "reaper";
            /**
             * F0 Frame Interval Ms
             * @default 1
             */
            f0_frame_interval_ms: number;
            /**
             * Frame Length
             * @default 128
             */
            frame_length: number;
            /**
             * Frame Shift
             * @default 32
             */
            frame_shift: number;
            /**
             * Lpc Order
             * @default 20
             */
            lpc_order: number;
            /**
             * Max F0 Hz
             * @default 300
             */
            max_f0_hz: number;
            /**
             * Min F0 Hz
             * @default 50
             */
            min_f0_hz: number;
            /**
             * Negative Peak Threshold
             * @default -0.005
             */
            negative_peak_threshold: number;
            /**
             * Preemphasis
             * @default 0.98
             */
            preemphasis: number;
            /**
             * Pulse Inner Periods
             * @default 0.5
             */
            pulse_inner_periods: number;
            /**
             * Pulse Outer Periods
             * @default 1.5
             */
            pulse_outer_periods: number;
            /**
             * Silence Padding Ms
             * @default 8
             */
            silence_padding_ms: number;
            /**
             * Silence Threshold Db
             * @default -45
             */
            silence_threshold_db: number;
            /**
             * Target Sample Rate
             * @default 11025
             * @constant
             */
            target_sample_rate: 11025;
            /**
             * Trim Silence
             * @default true
             */
            trim_silence: boolean;
            /**
             * Voiced Margin Ms
             * @default 30
             */
            voiced_margin_ms: number;
            /**
             * Window Name
             * @default hamming
             * @enum {string}
             */
            window_name: "hamming" | "hann" | "blackman" | "rectangular";
        };
        /** M07Controls */
        M07Controls: {
            /** Axis */
            axis: number[];
            /** Source */
            source: number[];
            /** Target */
            target: number[];
        };
        /** M07Generation */
        M07Generation: {
            /**
             * Energy Match
             * @default true
             */
            energy_match: boolean;
            /**
             * Normalize To Source
             * @default true
             */
            normalize_to_source: boolean;
            /**
             * Output Peak Limit
             * @default 0.98
             */
            output_peak_limit: number;
            /**
             * Step Count
             * @default 9
             */
            step_count: number;
        };
        /** M07Manifest */
        M07Manifest: {
            /**
             * Complete
             * @default true
             * @constant
             */
            complete: true;
            /** Core Version */
            core_version: string;
            /** Files */
            files: components["schemas"]["AcousticManagedFile"][];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "managed_m07_files";
            /**
             * Operation
             * @default phonation_synthesis
             * @constant
             */
            operation: "phonation_synthesis";
            /**
             * Policy Version
             * @default 1
             * @enum {integer}
             */
            policy_version: 1 | 2;
        };
        /** M07Request */
        M07Request: {
            /**
             * Action
             * @enum {string}
             */
            action: "analyze" | "apply" | "generate";
            /**
             * Alignment
             * @default normalize
             * @enum {string}
             */
            alignment: "normalize" | "onset";
            analysis?: components["schemas"]["M07Analysis"];
            /** Analysis Job Id */
            analysis_job_id?: string | null;
            /**
             * Batch Group Count
             * @default 1
             * @enum {integer}
             */
            batch_group_count: 1 | 6;
            /**
             * Batch Group Index
             * @default 0
             */
            batch_group_index: number;
            /** Batch Id */
            batch_id?: string | null;
            /**
             * Continuum Type
             * @default 2
             * @enum {integer}
             */
            continuum_type: 1 | 2 | 3;
            controls?: components["schemas"]["M07Controls"] | null;
            generation?: components["schemas"]["M07Generation"];
            /** Idempotency Key */
            idempotency_key: string;
            /**
             * Point Count
             * @default 21
             */
            point_count: number;
            /** Project Id */
            project_id: string;
            /**
             * Reverse Direction
             * @default false
             */
            reverse_direction: boolean;
            /**
             * Schema Version
             * @default m07/1
             * @constant
             */
            schema_version: "m07/1";
            source: components["schemas"]["AcousticAssetRef"];
            target: components["schemas"]["AcousticAssetRef"];
        };
        /** M08Config */
        M08Config: {
            /**
             * Action
             * @enum {string}
             */
            action: "preview" | "synthesize" | "transform" | "linear";
            /** End */
            end?: number | null;
            /** Modified F0 */
            modified_f0?: number[] | null;
            /**
             * Offset
             * @default false
             */
            offset: boolean;
            /**
             * Pitch Hz
             * @default 0
             */
            pitch_hz: number;
            /**
             * Pitch Ratio
             * @default 1
             */
            pitch_ratio: number;
            /** Points */
            points?: components["schemas"]["M08Point"][];
            /**
             * Speed
             * @default 1
             */
            speed: number;
            /**
             * Start
             * @default 0
             */
            start: number;
        };
        /** M08Manage */
        M08Manage: {
            /** Ids */
            ids: string[];
            /** Names */
            names?: string[];
            /** Project Id */
            project_id: string;
            source: components["schemas"]["AcousticAssetRef"];
        };
        /** M08Manifest */
        M08Manifest: {
            /** Aliases */
            aliases?: {
                [key: string]: string;
            };
            /**
             * Complete
             * @default true
             * @constant
             */
            complete: true;
            /** Core Version */
            core_version: string;
            /** Deleted */
            deleted?: string[];
            /** Files */
            files: components["schemas"]["AcousticManagedFile"][];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "managed_m08_files";
            /**
             * Operation
             * @default pitch_manipulation
             * @constant
             */
            operation: "pitch_manipulation";
            /**
             * Policy Version
             * @default 1
             * @enum {integer}
             */
            policy_version: 1 | 2;
            /** Saved */
            saved?: string[];
        };
        /** M08Point */
        M08Point: {
            /** Freqs */
            freqs: number[];
            /**
             * Mode
             * @default order
             * @enum {string}
             */
            mode: "full" | "order" | "reverse" | "constant";
            /** Time */
            time: number;
        };
        /** M08Request */
        M08Request: {
            audio: components["schemas"]["AcousticAssetRef"];
            config: components["schemas"]["M08Config"];
            /** Idempotency Key */
            idempotency_key: string;
            /** Project Id */
            project_id: string;
            /**
             * Schema Version
             * @default m08/1
             * @constant
             */
            schema_version: "m08/1";
        };
        /** M08Source */
        M08Source: {
            /** Project Id */
            project_id: string;
            source: components["schemas"]["AcousticAssetRef"];
        };
        /** M11Config */
        M11Config: {
            /**
             * Beam
             * @default 10
             */
            beam: number;
            /**
             * Retry Beam
             * @default 40
             */
            retry_beam: number;
        };
        /** M11CorpusItem */
        M11CorpusItem: {
            audio: components["schemas"]["AcousticAssetRef"];
            /** Name */
            name: string;
            transcript: components["schemas"]["AcousticAssetRef"];
            /**
             * Transcript Format
             * @default .lab
             * @enum {string}
             */
            transcript_format: ".lab" | ".txt" | ".TextGrid";
        };
        /** M11Manifest */
        M11Manifest: {
            /**
             * Complete
             * @default true
             * @constant
             */
            complete: true;
            /** Core Version */
            core_version: string;
            /** Files */
            files: components["schemas"]["AcousticManagedFile"][];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "managed_m11_files";
            /**
             * Operation
             * @default mfa_alignment
             * @constant
             */
            operation: "mfa_alignment";
            /**
             * Policy Version
             * @default 1
             * @enum {integer}
             */
            policy_version: 1 | 2;
        };
        /** M11Request */
        M11Request: {
            config?: components["schemas"]["M11Config"];
            /** Corpus */
            corpus: components["schemas"]["M11CorpusItem"][];
            dictionary?: components["schemas"]["AcousticAssetRef"] | null;
            /** Idempotency Key */
            idempotency_key: string;
            /** Model Id */
            model_id: string;
            /** Project Id */
            project_id: string;
            /** Runtime Id */
            runtime_id: string;
            /**
             * Schema Version
             * @default m11/1
             * @constant
             */
            schema_version: "m11/1";
        };
        /** M14Config */
        M14Config: {
            /**
             * Action
             * @enum {string}
             */
            action: "preview" | "export";
            /**
             * Consonant Only As Zero Initial
             * @default true
             */
            consonant_only_as_zero_initial: boolean;
            font?: components["schemas"]["FigureFontSnapshot"] | null;
            settings?: components["schemas"]["M14Settings"] | null;
            /**
             * Skip First Row
             * @default true
             */
            skip_first_row: boolean;
        };
        /** M14Manifest */
        M14Manifest: {
            /**
             * Complete
             * @default true
             * @constant
             */
            complete: true;
            /** Core Version */
            core_version: string;
            /** Files */
            files: components["schemas"]["AcousticManagedFile"][];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "managed_m14_files";
            /**
             * Operation
             * @default phonology_induction
             * @constant
             */
            operation: "phonology_induction";
            /**
             * Policy Version
             * @default 1
             * @enum {integer}
             */
            policy_version: 1 | 2;
        };
        /** M14Request */
        M14Request: {
            config: components["schemas"]["M14Config"];
            /** Idempotency Key */
            idempotency_key: string;
            /** Project Id */
            project_id: string;
            /**
             * Schema Version
             * @default m14/1
             * @constant
             */
            schema_version: "m14/1";
            table: components["schemas"]["AcousticAssetRef"];
        };
        /** M14Settings */
        M14Settings: {
            /** Final Map */
            final_map?: {
                [key: string]: string;
            };
            /** Final Order */
            final_order: string[];
            /** Initial Map */
            initial_map?: {
                [key: string]: string;
            };
            /** Initial Order */
            initial_order: string[];
            /** Tone Map */
            tone_map: {
                [key: string]: string;
            };
            /** Tone Order */
            tone_order: string[];
        };
        /** ParameterTable */
        ParameterTable: {
            /** Columns */
            columns: string[];
            /** Kinds */
            kinds: ("number" | "text")[];
            /** Rows */
            rows: (number | string | null)[][];
            /**
             * Schema Version
             * @default m02/1
             * @constant
             */
            schema_version: "m02/1";
            /** Sha256 */
            sha256: string;
        };
        /** PreviewInterval */
        PreviewInterval: {
            /** Text */
            text: string;
            /** Xmax */
            xmax: number;
            /** Xmin */
            xmin: number;
        };
        /** PreviewTier */
        PreviewTier: {
            /** Intervals */
            intervals: components["schemas"]["PreviewInterval"][];
            /** Name */
            name: string;
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
            manifest: components["schemas"]["JobManifest"] | components["schemas"]["FileManifest"] | components["schemas"]["AcousticFileManifest"] | components["schemas"]["AcousticTaskManifest"] | components["schemas"]["Spec2WavManifest"] | components["schemas"]["EggManifest"] | components["schemas"]["LpcManifest"] | components["schemas"]["M08Manifest"] | components["schemas"]["M14Manifest"] | components["schemas"]["M06Manifest"] | components["schemas"]["M07Manifest"] | components["schemas"]["M11Manifest"] | components["schemas"]["M05Manifest"];
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
        /** Spec2WavConfig */
        Spec2WavConfig: {
            /** Corners */
            corners?: components["schemas"]["ImagePoint"][] | null;
            /**
             * Freq End
             * @default 11025
             */
            freq_end: number;
            /**
             * Freq Start
             * @default 0
             */
            freq_start: number;
            /**
             * Max Db
             * @default 0
             */
            max_db: number;
            /**
             * Min Db
             * @default -30
             */
            min_db: number;
            /**
             * N Iter
             * @default 32
             */
            n_iter: number;
            /**
             * Seed
             * @default 0
             */
            seed: number;
            /**
             * Target Sr
             * @default 44100
             * @enum {integer}
             */
            target_sr: 0 | 8000 | 16000 | 22050 | 24000 | 32000 | 44100 | 48000 | 96000;
            /**
             * Time End
             * @default 1
             */
            time_end: number;
            /**
             * Time Start
             * @default 0
             */
            time_start: number;
            /**
             * Win Length Ms
             * @default 10
             */
            win_length_ms: number;
        };
        /** Spec2WavManifest */
        Spec2WavManifest: {
            /**
             * Complete
             * @default true
             * @constant
             */
            complete: true;
            /** Core Version */
            core_version: string;
            /** Files */
            files: components["schemas"]["AcousticManagedFile"][];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "managed_spec2wav_files";
            /**
             * Operation
             * @default spectrogram_to_audio
             * @constant
             */
            operation: "spectrogram_to_audio";
            /**
             * Policy Version
             * @default 1
             * @enum {integer}
             */
            policy_version: 1 | 2;
        };
        /** Spec2WavRequest */
        Spec2WavRequest: {
            config: components["schemas"]["Spec2WavConfig"];
            /** Idempotency Key */
            idempotency_key: string;
            image: components["schemas"]["AcousticAssetRef"];
            /** Project Id */
            project_id: string;
            /**
             * Schema Version
             * @default m09/1
             * @constant
             */
            schema_version: "m09/1";
        };
        /** SpectrogramPreview */
        SpectrogramPreview: {
            /** Backend */
            backend: string;
            /** Dx */
            dx: number;
            /** Dy */
            dy: number;
            /** Dynamic Range */
            dynamic_range: number;
            /** End */
            end: number;
            /** Frequency Max */
            frequency_max: number;
            /** Height */
            height: number;
            /** Parselmouth Version */
            parselmouth_version: string;
            /** Pixels Base64 */
            pixels_base64: string;
            /** Praat Version */
            praat_version: string;
            /** Preemphasis */
            preemphasis: number;
            /** Sha256 */
            sha256: string;
            /** Start */
            start: number;
            /** Time Step */
            time_step: number;
            /** Width */
            width: number;
            /** Window Length */
            window_length: number;
            /** X1 */
            x1: number;
            /** Y1 */
            y1: number;
        };
        /** StorageUsage */
        StorageUsage: {
            /** Available Bytes */
            available_bytes: number;
            /** Frozen */
            frozen: boolean;
            /**
             * Over Quota
             * @default false
             */
            over_quota: boolean;
            /**
             * Policy Version
             * @default 1
             * @enum {integer}
             */
            policy_version: 1 | 2;
            /** Quota Bytes */
            quota_bytes: number;
            /** Ready */
            ready: boolean;
            /** Reserved Bytes */
            reserved_bytes: number;
            /**
             * Retention Seconds
             * @default 604800
             */
            retention_seconds: number;
            /** Used Bytes */
            used_bytes: number;
        };
        /** TextGridPreview */
        TextGridPreview: {
            /** Asset Id */
            asset_id: string;
            /** Sha256 */
            sha256: string;
            /** Tiers */
            tiers: components["schemas"]["PreviewTier"][];
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
    asset_parameter_table: {
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
                    "application/json": components["schemas"]["ParameterTable"];
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
    asset_spectrogram_preview: {
        parameters: {
            query: {
                channel: number;
                start: number;
                end: number;
                width?: number;
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
                    "application/json": components["schemas"]["SpectrogramPreview"];
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
    preview_textgrid: {
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
                    "application/json": components["schemas"]["TextGridPreview"];
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
    create_acoustic_batch: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["BatchRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["BatchView"];
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
    list_acoustic_batches: {
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
                    "application/json": components["schemas"]["BatchList"];
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
    get_acoustic_batch: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                batch_id: string;
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
                    "application/json": components["schemas"]["BatchView"];
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
    cancel_acoustic_batch: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                batch_id: string;
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
                    "application/json": components["schemas"]["BatchView"];
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
    create_egg_job: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["EggRequest"];
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
    check_egg_export_fonts: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["FigureFontSnapshot"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["FontPreflight"];
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
    register_local_acoustic_input: {
        parameters: {
            query: {
                role: string;
                name: string;
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
                    "application/json": unknown;
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
    convert_local_legacy_lip: {
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
                    "application/json": unknown;
                };
            };
        };
    };
    read_local_acoustic_result: {
        parameters: {
            query?: {
                offset?: number;
                size?: number;
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
                    "application/json": unknown;
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
    create_lpc_job: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["LpcRequest"];
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
    check_lpc_export_fonts: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["FigureFontSnapshot"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["FontPreflight"];
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
    lip_catalog: {
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
                    "application/json": unknown;
                };
            };
        };
    };
    create_lip_job: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["M05Request"];
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
    begin_local_lip_video: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["M05Upload"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
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
    write_local_lip_video: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                key: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["M05Block"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
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
    abort_local_lip_video: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                key: string;
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
                    "application/json": unknown;
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
    finalize_local_lip_video: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                key: string;
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
                    "application/json": unknown;
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
    render_lip_animation: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                key: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["M05Config"];
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
    create_speech_synthesis_job: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["M06Request"];
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
    create_phonation_synthesis_job: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["M07Request"];
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
    create_m08_job: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["M08Request"];
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
    m08_history: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["M08Source"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
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
    list_m08_results: {
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
                    "application/json": unknown;
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
    remove_m08_results: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["M08Manage"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
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
    rename_m08_results: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["M08Manage"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
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
    save_m08_result: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["M08Manage"];
            };
        };
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
    mfa_component_catalog: {
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
                    "application/json": unknown;
                };
            };
        };
    };
    manage_local_mfa_component: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["LocalM11ComponentRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
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
    create_mfa_alignment_job: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["M11Request"];
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
    create_phonology_job: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["M14Request"];
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
    find_acoustic_parent: {
        parameters: {
            query: {
                project_id: string;
                sha256: string;
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
                    "application/json": unknown;
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
    create_spec2wav_job: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Spec2WavRequest"];
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
    local_parameter_table: {
        parameters: {
            query: {
                name: string;
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
                    "application/json": components["schemas"]["ParameterTable"];
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
    local_spectrogram_preview: {
        parameters: {
            query: {
                channel: number;
                start: number;
                end: number;
                width?: number;
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
                    "application/json": components["schemas"]["SpectrogramPreview"];
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
