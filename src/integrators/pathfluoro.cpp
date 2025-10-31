#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-pathfluoro:

Path tracer with fluorescence (:monosp:`pathfluoro`)
----------------------------

.. pluginparameters::

 * - max_depth
   - |int|
   - Specifies the longest path depth in the generated output image (where -1
     corresponds to :math:`\infty`). A value of 1 will only render directly
     visible light sources. 2 will lead to single-bounce (direct-only)
     illumination, and so on. (Default: -1)

 * - rr_depth
   - |int|
   - Specifies the path depth, at which the implementation will begin to use
     the *russian roulette* path termination criterion. For example, if set to
     1, then path generation may randomly cease after encountering directly
     visible surfaces. (Default: 5)

 * - hide_emitters
   - |bool|
   - Hide directly visible emitters. (Default: no, i.e. |false|)

This integrator is based on the standard path tracer, but supports fluorescent
(wavelength-shifting) BRDFs.

Compared to the standard path tracer, at every intersection, this integrator
implementation additionally tests if the interaction between light and material
was fluorescent. If so, the integrator applies a wavelength shift to the next
ray, with the new wavelength sampled from the material's excitation spectrum.
The new wavelength is also used to sample scene emitters. However, the spectral
appearance of the object is determined by interaction with the original wavelength.

Like the :ref:`direct <integrator-direct>` plugin, the path tracer internally
relies on multiple importance sampling to combine BSDF and emitter samples. The
main difference in comparison to the former plugin is that it considers light
paths of arbitrary length to compute both direct and indirect illumination.

.. note:: This integrator does not handle participating media

.. tabs::
    .. code-tab::  xml
        :name: pathfluoro-integrator

        <integrator type="pathfluoro">
            <integer name="max_depth" value="8"/>
        </integrator>

    .. code-tab:: python

        'type': 'pathfluoro',
        'max_depth': 8

 */

template <typename Float, typename Spectrum>
class PathFluoroIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    PathFluoroIntegrator(const Properties &props) : Base(props) {
        if constexpr (!is_spectral_v<Spectrum>) {
            Log(Error, "This integrator can only be used in Mitsuba variants that "
                       "perform a spectral simulation.");
        }
    }

    std::pair<Spectrum, Bool> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Bool active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        if (unlikely(m_max_depth == 0))
            return { 0.f, false };

        // --------------------- Configure loop state ----------------------

        Ray3f ray                     = Ray3f(ray_);
        Spectrum throughput           = 1.f;
        Spectrum result               = 0.f;
        Float eta                     = 1.f;
        PreliminaryIntersection3f pi  = dr::zeros<PreliminaryIntersection3f>();
        UInt32 depth                  = 0;

        // If m_hide_emitters == false, the environment emitter will be visible
        Mask valid_ray = !m_hide_emitters && (scene->environment() != nullptr);

        // Variables caching information from the previous bounce
        Interaction3f prev_si         = dr::zeros<Interaction3f>();
        Float         prev_bsdf_pdf   = 1.f;
        Bool          prev_bsdf_delta = true;
        BSDFContext   bsdf_ctx;

        /* Set up a Dr.Jit loop. This optimizes away to a normal loop in scalar
           mode, and it generates either a megakernel (default) or
           wavefront-style renderer in JIT variants. This can be controlled by
           passing the '-W' command line flag to the mitsuba binary or
           enabling/disabling the JitFlag.LoopRecord bit in Dr.Jit.
        */
        struct LoopState {
            Ray3f ray;
            PreliminaryIntersection3f pi;
            Spectrum throughput;
            Spectrum result;
            Float eta;
            UInt32 depth;
            Mask valid_ray;
            Interaction3f prev_si;
            Float prev_bsdf_pdf;
            Bool prev_bsdf_delta;
            Bool active;
            Sampler* sampler;

            DRJIT_STRUCT(LoopState, ray, pi, throughput, result, eta, depth, \
                valid_ray, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                active, sampler)
        } ls = {
            ray,
            pi,
            throughput,
            result,
            eta,
            depth,
            valid_ray,
            prev_si,
            prev_bsdf_pdf,
            prev_bsdf_delta,
            active,
            sampler
        };

        // First bounce is usually coherent - don't reorder threads
        ls.pi = scene->ray_intersect_preliminary(ls.ray,
                                                 /* coherent = */ true,
                                                 /* reorder = */ false,
                                                 /* reorder_hint = */ 0,
                                                 /* reorder_hint_bits = */ 0,
                                                 ls.active);

        // ---------------------- Hide area emitters ----------------------

        /* dr::any_or() checks for active entries in the provided boolean
           array. JIT/Megakernel modes can't do this test efficiently as
           each Monte Carlo sample runs independently. In this case,
           dr::any_or<..>() returns the template argument (true) which means
           that the 'if' statement is always conservatively taken. */

        if (m_hide_emitters && dr::any_or<true>(ls.depth == 0u)) {
            // Did we hit an area emitter? If so, skip all area emitters along this ray
            Mask skip_emitters = ls.pi.is_valid() &&
                                 (ls.pi.shape->emitter() != nullptr) &&
                                 ls.active;

            if (dr::any_or<true>(skip_emitters)) {
                SurfaceInteraction3f si = ls.pi.compute_surface_interaction(
                    ls.ray, +RayFlags::Minimal, skip_emitters);
                Ray3f ray = si.spawn_ray(ls.ray.d);
                PreliminaryIntersection3f pi_after_skip =
                    Base::skip_area_emitters(scene, ray, true, skip_emitters);
                dr::masked(ls.pi, skip_emitters) = pi_after_skip;
            }
        }

        dr::tie(ls) = dr::while_loop(dr::make_tuple(ls),
            [](const LoopState& ls) { return ls.active; },
            [this, scene, bsdf_ctx](LoopState& ls) {

            /* dr::while_loop implicitly masks all code in the loop using the
               'active' flag, so there is no need to pass it to every function */

            // Fill out all information of the interaction
            SurfaceInteraction3f si =
                ls.pi.compute_surface_interaction(ls.ray, +RayFlags::All);

            // ---------------------- Direct emission ----------------------

            if (dr::any_or<true>(si.emitter(scene) != nullptr)) {
                DirectionSample3f ds(scene, si, ls.prev_si);
                Float em_pdf = 0.f;

                if (dr::any_or<true>(!ls.prev_bsdf_delta))
                    em_pdf = scene->pdf_emitter_direction(ls.prev_si, ds,
                                                          !ls.prev_bsdf_delta);

                // Compute MIS weight for emitter sample from previous bounce
                Float mis_bsdf = mis_weight(ls.prev_bsdf_pdf, em_pdf);

                // Accumulate, being careful with polarization (see spec_fma)
                ls.result = spec_fma(
                    ls.throughput,
                    ds.emitter->eval(si, ls.prev_bsdf_pdf > 0.f) * mis_bsdf,
                    ls.result);
            }

            // Continue tracing the path at this point?
            Bool active_next = (ls.depth + 1 < m_max_depth) && si.is_valid();

            if (dr::none_or<false>(active_next)) {
                ls.active = active_next;
                ls.valid_ray |= (si.emitter(scene) != nullptr) && !m_hide_emitters;
                return; // early exit for scalar mode
            }

            BSDFPtr bsdf = si.bsdf(ls.ray);

            /* Separate sample() from eval_pdf() to before emitter sampling
               to decide if the interaction should be wavelength-shifting
               i.e. fluorescent */
            Float sample_1 = ls.sampler->next_1d();
            Point2f sample_2 = ls.sampler->next_2d();

            auto [bsdf_sample, bsdf_weight] =
                bsdf->sample(bsdf_ctx, si, sample_1, sample_2);

            /* If a fluorescent component is sampled, sample the excitation
               distribution to shift the wavelength of the incoming ray. The
               new wavelength is used to sample emitters and for the next ray,
               but the original wavelength is used to evaluate the fluorescent
               emission of the hit object itself. */
            Mask is_fluoro = has_flag(bsdf_sample.sampled_type,
                                      BSDFFlags::FluorescentReflection);
            SurfaceInteraction3f shifted_si = si;
            UnpolarizedSpectrum excite_weight (0.f);
            std::tie(shifted_si.wavelengths, excite_weight) =
                bsdf->sample_excitation(si, sample_1);
            shifted_si = dr::select(is_fluoro, shifted_si, si);
            /* For fluorescent materials, the fluorescent emission distribution
               is normalised to have a unit integral, while the excitation
               distribution represents the proportion of the incoming energy
               that is re-emitted. So, the BSDF weight needs to be multiplied
               by the excitation weight. */
            bsdf_weight = dr::select(is_fluoro, bsdf_weight * excite_weight, bsdf_weight);

            // ---------------------- Emitter sampling ----------------------

            // Perform emitter sampling?
            Mask active_em = active_next && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            DirectionSample3f ds = dr::zeros<DirectionSample3f>();
            Spectrum em_weight = dr::zeros<Spectrum>();
            Vector3f wo = dr::zeros<Vector3f>();

            if (dr::any_or<true>(active_em)) {
                // Sample the emitter
                std::tie(ds, em_weight) = scene->sample_emitter_direction(
                    shifted_si, ls.sampler->next_2d(), true, active_em);
                active_em &= (ds.pdf != 0.f);

                /* Given the detached emitter sample, recompute its contribution
                   with AD to enable light source optimization. */
                if (dr::grad_enabled(si.p)) {
                    ds.d = dr::normalize(ds.p - si.p);
                    Spectrum em_val = scene->eval_emitter_direction(shifted_si, ds, active_em);
                    em_weight = dr::select(ds.pdf != 0, em_val / ds.pdf, 0);
                }

                wo = si.to_local(ds.d);
            }

            // ------ Evaluate BSDF * cos(theta) and sample direction -------

            auto [bsdf_val, bsdf_pdf] = dr::select(
                is_fluoro,
                bsdf->eval_pdf_fluoro(bsdf_ctx, si, wo),
                bsdf->eval_pdf(bsdf_ctx, si, wo)
            );

            // --------------- Emitter sampling contribution ----------------

            if (dr::any_or<true>(active_em)) {
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                // Compute the MIS weight
                Float mis_em =
                    dr::select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                // Accumulate, being careful with polarization (see spec_fma)
                Spectrum value = dr::select(
                    is_fluoro,
                    excite_weight * bsdf_val * em_weight * mis_em,
                    bsdf_val * em_weight * mis_em
                );
                ls.result[active_em] = spec_fma(ls.throughput, value, ls.result);
            }

            // ---------------------- BSDF sampling ----------------------

            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);

            ls.ray = shifted_si.spawn_ray(si.to_world(bsdf_sample.wo));

            /* When the path tracer is differentiated, we must be careful that
               the generated Monte Carlo samples are detached (i.e. don't track
               derivatives) to avoid bias resulting from the combination of moving
               samples and discontinuous visibility. We need to re-evaluate the
               BSDF differentiably with the detached sample in that case. */
            if (dr::grad_enabled(ls.ray)) {
                ls.ray = dr::detach<true>(ls.ray);

                // Recompute 'wo' to propagate derivatives to cosine term
                Vector3f wo_2 = si.to_local(ls.ray.d);
                auto [bsdf_val_2, bsdf_pdf_2] = dr::select(
                    is_fluoro,
                    bsdf->eval_pdf_fluoro(bsdf_ctx, si, wo_2, ls.active),
                    bsdf->eval_pdf(bsdf_ctx, si, wo_2, ls.active)
                );

                bsdf_weight[bsdf_pdf_2 > 0.f] = bsdf_val_2 / dr::detach(bsdf_pdf_2);
            }

            // ------ Update loop variables based on current interaction ------

            ls.throughput *= bsdf_weight;
            ls.eta *= bsdf_sample.eta;
            ls.valid_ray |= ls.active && si.is_valid() &&
                            !has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);

            // Information about the current vertex needed by the next iteration
            ls.prev_si = Interaction3f(shifted_si);
            ls.prev_bsdf_pdf = bsdf_sample.pdf;
            ls.prev_bsdf_delta = has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta);

            // -------------------- Stopping criterion ---------------------

            dr::masked(ls.depth, si.is_valid()) += 1;

            Float throughput_max = dr::max(unpolarized_spectrum(ls.throughput));

            Float rr_prob = dr::minimum(throughput_max * dr::square(ls.eta), .95f);
            Mask rr_active = ls.depth >= m_rr_depth,
                 rr_continue = ls.sampler->next_1d() < rr_prob;

            /* Differentiable variants of the renderer require the russian
               roulette sampling weight to be detached to avoid bias. This is a
               no-op in non-differentiable variants. */
            ls.throughput[rr_active] *= dr::rcp(dr::detach(rr_prob));

            ls.active = active_next && (!rr_active || rr_continue) &&
                        (throughput_max != 0.f);

            // Reorder threads based on the shape they hit
            ls.pi = scene->ray_intersect_preliminary(ls.ray,
                                                     /* coherent = */ false,
                                                     /* reorder = */ jit_flag(JitFlag::LoopRecord),
                                                     /* reorder_hint = */ 0,
                                                     /* reorder_hint_bits = */ 0,
                                                     ls.active);
        });

        return {
            /* spec  = */ dr::select(ls.valid_ray, ls.result, 0.f),
            /* valid = */ ls.valid_ray
        };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("PathFluoroIntegrator[\n"
            "  max_depth = %u,\n"
            "  rr_depth = %u\n"
            "]", m_max_depth, m_rr_depth);
    }

    /// Compute a multiple importance sampling weight using the power heuristic
    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::detach<true>(dr::select(dr::isfinite(w), w, 0.f));
    }

    /**
     * \brief Perform a Mueller matrix multiplication in polarized modes, and a
     * fused multiply-add otherwise.
     */
    Spectrum spec_fma(const Spectrum &a, const Spectrum &b,
                      const Spectrum &c) const {
        if constexpr (is_polarized_v<Spectrum>)
            return a * b + c;
        else
            return dr::fmadd(a, b, c);
    }

    MI_DECLARE_CLASS(PathFluoroIntegrator)
};

MI_EXPORT_PLUGIN(PathFluoroIntegrator)
NAMESPACE_END(mitsuba)
