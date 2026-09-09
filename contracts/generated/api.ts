/** Generated from contracts/openapi.json. Do not edit. */
export interface paths {
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
}
export type webhooks = Record<string, never>;
export interface components {
    schemas: {
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
             * @default 1.0.0
             */
            api_version: string;
            /** Limitations */
            limitations: string[];
            /**
             * Stage
             * @default P02
             * @constant
             */
            stage: "P02";
        };
        /** Health */
        Health: {
            /**
             * Api Version
             * @default 1.0.0
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
}
