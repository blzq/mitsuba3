#include <mitsuba/render/volume.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/interaction.h>

NAMESPACE_BEGIN(mitsuba)

// =======================================================================
//! Volume base implementation
// =======================================================================

MI_VARIANT Volume<Float, Spectrum>::Volume(const Properties &props)
    : JitObject<Volume>(props.id()) {
    m_to_local = props.get<ScalarAffineTransform4f>("to_world", ScalarAffineTransform4f()).inverse();
    m_channel_count = 0;
    update_bbox();
}

MI_VARIANT typename Volume<Float, Spectrum>::UnpolarizedSpectrum
Volume<Float, Spectrum>::eval(const Interaction3f &, Mask) const {
    NotImplementedError("eval");
}

MI_VARIANT Float Volume<Float, Spectrum>::eval_1(const Interaction3f &, Mask) const {
    NotImplementedError("eval_1");
}

MI_VARIANT typename Volume<Float, Spectrum>::Vector3f
Volume<Float, Spectrum>::eval_3(const Interaction3f &, Mask) const {
    NotImplementedError("eval_3");
}

MI_VARIANT dr::Array<Float, 6>
Volume<Float, Spectrum>::eval_6(const Interaction3f &, Mask) const {
    NotImplementedError("eval_6");
}

MI_VARIANT void
Volume<Float, Spectrum>::eval_n(const Interaction3f & /*it*/,
                                Float * /*out*/,
                                Mask /*active*/) const {
    NotImplementedError("eval_n");
}

MI_VARIANT std::pair<typename Volume<Float, Spectrum>::UnpolarizedSpectrum,
                     typename Volume<Float, Spectrum>::Vector3f>
Volume<Float, Spectrum>::eval_gradient(const Interaction3f & /*it*/, Mask /*active*/) const {
    NotImplementedError("eval_gradient");
}

MI_VARIANT std::pair<typename Volume<Float, Spectrum>::Wavelength,
                     typename Volume<Float, Spectrum>::UnpolarizedSpectrum>
Volume<Float, Spectrum>::sample_spectrum(const Interaction3f &_it,
                                         const Wavelength &sample,
                                         Mask active) const {
    MI_MASKED_FUNCTION(ProfilerPhase::TextureSample, active);

    if (dr::none_or<false>(active))
        return { dr::zeros<Wavelength>(), dr::zeros<UnpolarizedSpectrum>() };

    if constexpr (is_spectral_v<Spectrum>) {
        Interaction3f it(_it);
        // Uniform sample - not proportional to local spectral distribution
        it.wavelengths = MI_CIE_MIN + (MI_CIE_MAX - MI_CIE_MIN) * sample;
        // importance weight accounts for uniform sample
        return { it.wavelengths,
                 eval(it, active) * (MI_CIE_MAX - MI_CIE_MIN) };
    } else {
        DRJIT_MARK_USED(sample);
        UnpolarizedSpectrum value = eval(_it, active);
        return { dr::empty<Wavelength>(), value };
    }
}

MI_VARIANT typename Volume<Float, Spectrum>::ScalarFloat
Volume<Float, Spectrum>::max() const { NotImplementedError("max"); }

MI_VARIANT void
Volume<Float, Spectrum>::max_per_channel(ScalarFloat * /*out*/) const {
    NotImplementedError("max_per_channel");
}

MI_VARIANT typename Volume<Float, Spectrum>::ScalarVector3i
Volume<Float, Spectrum>::resolution() const {
    return ScalarVector3i(1, 1, 1);
}

// =======================================================================

MI_INSTANTIATE_CLASS(Volume)
NAMESPACE_END(mitsuba)
